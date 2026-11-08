"""Fixed-model, document-level feature interventions for Text Lab.

No training, no independent passage classification, no attention-as-explanation.
RACE: suppress one contextual EDU content vector at the node_proj input,
including its fixed-denominator contribution to ALL descendant-mean relation
nodes. Preserve node-type embeddings, edges, other EDU features and parameters.
RoBERTa: zero selected lexical word embeddings BEFORE contextual encoding;
keep positions, special tokens, sequence length and attention mask unchanged.
TF-IDF: suppress a segment's proportional share of the already-normalized
unigram/bigram feature vector. Keep vocabulary, IDF, normalizer and classifier.

These are reference-dependent sensitivity measurements, NOT authorship labels
or an additive decomposition of probabilities. Other contextual vectors and
RST structure can retain information about the suppressed unit.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import math
import re
import threading
import time
from typing import Any

import numpy as np
import torch

LABELS = ('human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized')
SCHEMA = 'ee6405.document-explanation/v1'
MAX_UNITS_PER_REQUEST = 8
MAX_ANALYSIS_SECONDS = 90.0  # checked between forwards, never kills model threads
PARITY_ATOL = 2e-5


class ExplanationError(ValueError):
    """A safe, explicit unsupported/misaligned explanation; prediction still works."""


def score_array(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = [value[k] for k in LABELS]
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (4,) or not np.isfinite(arr).all() or (arr < 0).any() or not np.isclose(arr.sum(), 1, atol=1e-5):
        raise ExplanationError('Invalid four-class score vector')
    return arr


def scores_dict(value: Any) -> dict[str, float]:
    return dict(zip(LABELS, map(float, score_array(value))))


def utf16_offset(text: str, index: int) -> int:
    return len(text[:index].encode('utf-16-le')) // 2


def span_record(text: str, start: int, end: int, ident: str) -> dict:
    if not 0 <= start < end <= len(text):
        raise ExplanationError('Invalid source-text span')
    return {'id': ident, 'start': start, 'end': end,
            'start_utf16': utf16_offset(text, start), 'end_utf16': utf16_offset(text, end),
            'status': 'not_computed', 'without_scores': None, 'delta_pp': None}


def display_segments(text: str) -> list[dict]:
    """Non-overlapping display groups, NOT claimed to be true discourse units."""
    # A delimiter is sentence punctuation followed by whitespace, or a blank line.
    # Partition includes intervening whitespace, preserving the exact source.
    ends = [m.end() for m in re.finditer(r'[.!?](?:[\"\u201d\u2019\']+)?(?=\s|$)|\n\s*\n', text)]
    ends.append(len(text))
    result, start = [], 0
    for end in sorted(set(ends)):
        if end > start and text[start:end].strip():
            result.append(span_record(text, start, end, f'u{len(result)+1}'))
            start = end
    if start < len(text) and result:
        result[-1]['end'] = len(text)
        result[-1]['end_utf16'] = utf16_offset(text, len(text))
    return result


def validate_units(units: list[dict], text: str) -> None:
    last = 0
    for unit in units:
        a, b = unit['start'], unit['end']
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text) or a < last:
            raise ExplanationError('Overlapping or invalid source-text unit spans')
        last = b


def token_indices(offsets: list, start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def check_parity(observed, expected, message='Explanation replay differs from document prediction') -> float:
    delta = float(np.max(np.abs(score_array(observed) - score_array(expected))))
    if delta > PARITY_ATOL:
        raise ExplanationError(f'{message} (max difference {delta:.7g}); refusing attribution')
    return delta


def race_suppressed_features(unified: torch.Tensor, type_zero: torch.Tensor,
                             graph: Any, nodes: list[dict], descendants: dict,
                             edu_id: int) -> tuple[torch.Tensor, list[int]]:
    """Suppress EDU content and all direct relation means, retaining type vectors.

    Works only on a single graph. Validates the exact relation initialization
    against the pinned forward contract before performing any intervention.
    Does not mutate unified or any parameter.
    """
    if unified.ndim != 2 or unified.shape[0] != graph.num_nodes:
        raise ExplanationError('Unexpected RACE node feature shape')
    mapping = graph._edu_mapping
    if str(edu_id) not in mapping:
        raise ExplanationError('Selected EDU is not present in the lexical graph')
    edu_slot = mapping[str(edu_id)]
    edu_row = graph._edu_indices[edu_slot]
    content = unified[edu_row] - type_zero
    changed = unified.clone()
    changed[edu_row] = type_zero
    touched = [edu_row]
    relations = [n for n in nodes if n['type'] == 'relation_node']
    if len(relations) != len(graph._relation_indices):
        raise ExplanationError('RACE relation order mismatch')
    for k, node in enumerate(relations):
        ids = descendants.get(node['id'], descendants.get(str(node['id']), []))
        present = [mapping[str(i)] for i in ids if str(i) in mapping]
        if edu_slot in present:
            # Match the exact pinned denominator, including any duplicates.
            row = graph._relation_indices[k]
            changed[row] -= content * (present.count(edu_slot) / len(present))
            touched.append(row)
    return changed, touched


def validate_relation_initialization(unified, type_embeddings, graph, nodes, descendants):
    edu = unified[graph._edu_indices] - type_embeddings[0]
    relations = [n for n in nodes if n['type'] == 'relation_node']
    for k, node in enumerate(relations):
        ids = descendants.get(node['id'], descendants.get(str(node['id']), []))
        selected = [graph._edu_mapping[str(i)] for i in ids if str(i) in graph._edu_mapping]
        expected = (edu[selected].mean(0) if selected else torch.zeros_like(type_embeddings[1])) + type_embeddings[1]
        actual = unified[graph._relation_indices[k]]
        if not torch.allclose(actual, expected, atol=5e-5, rtol=2e-5):
            raise ExplanationError('Relation features do not match the pinned descendant-mean contract')


@contextmanager
def temporary_hook(module, fn, *, before=False):
    handle = module.register_forward_pre_hook(fn) if before else module.register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


class Trace:
    """Capture numerical summaries of actual executed modules, not attention."""
    def __init__(self, model, root=None, race=False):
        self.model, self.root, self.race = model, root, race
        self.values, self.handles = {}, []
        self.unified = None

    def _keep(self, key, value):
        if not isinstance(value, torch.Tensor):
            return
        if value.ndim == 2 and self.root is not None and key != 'classifier_input':
            v = value[self.root].detach().clone()
        elif value.ndim == 3:
            v = value[0, 0].detach().clone()
        else:
            v = value.reshape(-1).detach().clone()
        self.values[key] = (list(value.shape), v)

    def __enter__(self):
        if self.race:
            def pre(_m, args):
                self.unified = args[0].detach().clone()
            self.handles.append(self.model.node_proj.register_forward_pre_hook(pre))
            self.handles.append(self.model.node_proj.register_forward_hook(lambda m, a, o: self._keep('node_projection', o)))
            for i, conv in enumerate(self.model.convs):
                self.handles.append(conv.register_forward_hook(lambda m, a, o, i=i: self._keep(f'graph_layer_{i+1}', o)))
        else:
            self.handles.append(self.model.bert_model.register_forward_hook(lambda m, a, o: self._keep('contextual_CLS', o.last_hidden_state)))
        self.handles.append(self.model.classifier.register_forward_pre_hook(lambda m, a: self._keep('classifier_input', a[0])))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def summaries(self, base=None):
        out = []
        for key, (shape, vector) in self.values.items():
            row = {'stage': key, 'tensor_shape': shape, 'vector_dimension': vector.numel(),
                   'rms': float(vector.float().square().mean().sqrt().item())}
            if base and key in base.values:
                v0 = base.values[key][1]
                row['change_rms'] = float((vector.float() - v0.float()).square().mean().sqrt().item())
            out.append(row)
        return out


def logits_and_scores(output):
    logits = output['logits'] if isinstance(output, dict) else output
    if not isinstance(logits, torch.Tensor) or logits.shape != (1, 4) or not torch.isfinite(logits).all():
        raise ExplanationError('Unexpected document model output')
    raw = logits[0].float().detach().cpu().numpy()
    probs = logits.float().softmax(-1)[0].detach().cpu().numpy()
    return raw, score_array(probs)


def tfidf_shares(vectorizer, text, units):
    """Allocate each feature occurrence to the segment containing its first word.

    An ngram spanning two display groups is assigned to the first-word group.
    Features repeated elsewhere retain those other occurrence shares. No re-fit
    or re-normalization is performed. Supports the project's fixed word model.
    """
    if (vectorizer.analyzer != 'word' or vectorizer.tokenizer is not None
            or vectorizer.preprocessor is not None or vectorizer.strip_accents is not None
            or vectorizer.stop_words is not None):
        raise ExplanationError('Unsupported TF-IDF preprocessing for exact occurrence mapping')
    pattern = re.compile(vectorizer.token_pattern)
    if pattern.groups:
        raise ExplanationError('Capturing token regex not supported')
    words = [(m.group().lower() if vectorizer.lowercase else m.group(), m.start()) for m in pattern.finditer(text)]
    occurrences = defaultdict(list)
    lo, hi = vectorizer.ngram_range
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            feature = ' '.join(x[0] for x in words[i:i+n])
            occurrences[feature].append(words[i][1])
    if Counter(vectorizer.build_analyzer()(text)) != Counter({k: len(v) for k, v in occurrences.items()}):
        raise ExplanationError('Tokenizer/preprocessing offset mapping cannot be proven')
    shares = {}
    for unit in units:
        values = {}
        for name, positions in occurrences.items():
            j = vectorizer.vocabulary_.get(name)
            if j is not None:
                count = sum(unit['start'] <= p < unit['end'] for p in positions)
                if count:
                    values[j] = count / len(positions)
        shares[unit['id']] = values
    return shares


def graph_description(model, graph, item, units):
    edus = [n for n in item['nodes'] if n['type'] == 'edu_node'][:len(item['edu_token_spans'])]
    rels = [n for n in item['nodes'] if n['type'] == 'relation_node']
    ordered = edus + rels
    if len(ordered) != graph.num_nodes:
        raise ExplanationError('Graph-node mapping differs from model graph')
    node_rows = []
    for i, n in enumerate(ordered):
        node_rows.append({'index': i, 'node_id': n['id'], 'kind': n['type'],
                          'relation': n.get('relation'), 'nuclearity': n.get('nuclearity'),
                          'unit_id': f"u{n['id']}" if n['type'] == 'edu_node' else None})
    edges = []
    names = list(model.edge_types_list)
    for i, (a, b) in enumerate(graph.edge_index.detach().cpu().T.tolist()):
        t = int(graph.edge_type[i])
        edges.append({'source': a, 'target': b, 'type_id': t, 'type': list(names[t])})
    lookup = {n['node_id']: n['index'] for n in node_rows}
    by_pair = {(e['source'], e['target']): e for e in edges}
    tree_edges = []
    for parent, child in item['parent_child_edges']:
        pair = (lookup.get(parent), lookup.get(child))
        if pair in by_pair:
            tree_edges.append(by_pair[pair])
    return {'nodes': node_rows, 'edges': edges, 'tree_edges': tree_edges, 'root_index': int(graph._root_idx),
            'layers': len(model.convs), 'edge_meaning': 'Actual typed edges; width is not importance.',
            'pooling': model.config.get('pooling_strategy', 'root')}


# ---- Preparation cache -------------------------------------------------
# Everything that depends only on (text, model) is built once and reused by
# later unit batches. Keys include the preprocessing version, so a change in
# tokenization, graph building or unit derivation can never reuse stale state.
PREPROCESSING_VERSION = 'ee6405.explain-prep/v1'
PREP_CACHE_ENTRIES = 4
DOC_CACHE_ENTRIES = 4
PREP_CACHE: 'OrderedDict[str, dict]' = OrderedDict()
DOC_CACHE: 'OrderedDict[str, dict]' = OrderedDict()
CACHE_LOCK = threading.Lock()
CACHE_STATS = {'hits': 0, 'misses': 0, 'evictions': 0, 'invalidations': 0}


def preparation_key(text: str, name: str, service: Any = None) -> str:
    """Version-scoped identity of one prepared document.

    The service instance is part of the identity because prepared state holds
    that instance's model objects and tensors; it must never cross instances.
    """
    owner = '' if service is None else str(id(service))
    return hashlib.sha256(f'{PREPROCESSING_VERSION}\x00{name}\x00{owner}\x00{text}'.encode('utf-8')).hexdigest()


def cache_stats() -> dict:
    with CACHE_LOCK:
        return {'prepared_entries': len(PREP_CACHE), 'prepared_capacity': PREP_CACHE_ENTRIES,
                'document_entries': len(DOC_CACHE), 'preprocessing_version': PREPROCESSING_VERSION,
                **CACHE_STATS}


def clear_preparation_cache() -> None:
    with CACHE_LOCK:
        PREP_CACHE.clear()
        DOC_CACHE.clear()
        CACHE_STATS['invalidations'] += 1


def reset_cache_stats() -> None:
    """Zero the counters without touching cached entries (used by tests)."""
    with CACHE_LOCK:
        for key in CACHE_STATS:
            CACHE_STATS[key] = 0


def forget_service(service: Any) -> None:
    """Drop cached state that belongs to one service instance (tests, restarts)."""
    with CACHE_LOCK:
        for key in [k for k, v in DOC_CACHE.items() if v['service'] is service]:
            DOC_CACHE.pop(key)
        PREP_CACHE.clear()


def remember_document(text: str, name: str, document: Any, service: Any = None) -> None:
    """Let /api/predict hand its result to the explanation path without a second forward."""
    if not isinstance(text, str) or not isinstance(document, dict):
        return
    key = preparation_key(text, name, service)
    with CACHE_LOCK:
        DOC_CACHE[key] = {'service': service, 'document': document}
        DOC_CACHE.move_to_end(key)
        while len(DOC_CACHE) > DOC_CACHE_ENTRIES:
            DOC_CACHE.popitem(last=False)


def _document_for(service, text: str, name: str) -> dict:
    key = preparation_key(text, name, service)
    with CACHE_LOCK:
        entry = DOC_CACHE.get(key)
        if entry is not None and entry['service'] is service:
            DOC_CACHE.move_to_end(key)
            return entry['document']
    document = service.predict(text, name)
    remember_document(text, name, document, service)
    return document


def _ensure_baseline(prep: dict, model, race: bool, graph, item) -> None:
    """Run the traced baseline forward once and validate the pinned contract."""
    if prep.get('base_trace') is not None:
        return
    root = graph._root_idx if race and graph is not None and graph._root_idx >= 0 else None
    with torch.inference_mode(), Trace(model, root, race) as base_trace:
        raw_base, replay = logits_and_scores(model(prep['batch']))
    check_parity(replay, prep['base_scores'])
    if race:
        if base_trace.unified is None:
            raise ExplanationError('RACE node projection hook was not executed')
        validate_relation_initialization(base_trace.unified, model.node_type_embedding.weight.detach(),
                                         graph, item['nodes'], item['descendant_map'])
    prep.update(base_trace=base_trace, raw_base=raw_base, replay=replay, stages=base_trace.summaries())


def _build_preparation(service, text: str, name: str, document: dict, baseline: bool = True) -> dict:
    """Tokenization, RST parse, graph build and (optionally) the baseline forward.

    The caller holds service.lock. Nothing here mutates weights or the graph.
    With baseline=False the expensive traced forward is deferred until the first
    influence request, so metadata-only preparation stays cheap.
    """
    base_scores = score_array(document['scores'])
    model, _ = service._load_model(name)
    encoded = service.tokenizer(text, max_length=512, truncation=True, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    covered = max((b for a, b in offsets if b > a), default=0)
    if not covered:
        raise ExplanationError('No lexical input tokens')
    # TF-IDF has no position embeddings, so it reads the whole text (see inference.py).
    if name == 'tfidf':
        covered = len(text)
    race = name.startswith('race')
    prep = {'document': document, 'base_scores': base_scores, 'model': model, 'encoded': encoded,
            'offsets': offsets, 'covered': covered, 'race': race, 'graph_json': None,
            'stages': [], 'item': None, 'graph': None, 'base_trace': None, 'batch': None,
            'x': None, 'shares': None, 'vectorizer': None, 'classifier': None,
            'raw_base': None, 'replay': None}
    graph = None
    graph_json = None
    if name == 'tfidf':
        units = display_segments(text)
        if list(model.named_steps) != ['tfidf', 'classifier']:
            raise ExplanationError('Unsupported TF-IDF pipeline')
        vectorizer, classifier = model.named_steps['tfidf'], model.named_steps['classifier']
        if list(classifier.classes_) != list(range(4)):
            raise ExplanationError('Classifier label order is not the main four-class order')
        x = vectorizer.transform([text[:covered]])
        shares = tfidf_shares(vectorizer, text[:covered], units)
        replay = classifier.predict_proba(x)[0]
        raw_base = np.asarray(classifier.decision_function(x)[0], dtype=float)
        check_parity(replay, base_scores)
        prep.update(x=x, shares=shares, vectorizer=vectorizer, classifier=classifier,
                    raw_base=raw_base, replay=replay,
                    stages=[{'stage': 'tfidf_vector', 'tensor_shape': list(x.shape), 'active_features': int(x.nnz)},
                            {'stage': 'linear_logits', 'tensor_shape': [1, 4]}],
                    method='tfidf_occurrence_share_suppression')
    else:
        if model.training or any(m.training for m in model.modules()):
            raise ExplanationError('Model must already be in eval mode; no mode is changed here')
        from experiments.neural import TextDataset, collate_batch
        record = {'item_id': 'explanation', 'article': text, 'label_id': -1}
        if race:
            from experiments.raw_graph import RawGraphDataset
            record['rst_structure'] = document.get('rst_tree')
            if not record['rst_structure']:
                raise ExplanationError('RACE did not return a real RST tree')
            item = RawGraphDataset([record], service.tokenizer)[0]
            graph = model.graph_builder.build({k: item[k] for k in ('nodes', 'parent_child_edges', 'root_node_ids')},
                                              max_edu_nodes=len(item['edu_token_spans']))
            if graph.num_nodes > 512:
                raise ExplanationError('Graph too large for bounded interactive explanation (512 nodes)')
            units = []
            edu_nodes = [n for n in item['nodes'] if n['type'] == 'edu_node']
            actual_visible = []
            for n in edu_nodes:
                unit = span_record(text, int(n['start']), int(n['end']), f"u{n['id']}")
                unit['node_id'] = n['id']
                inds = token_indices(offsets, unit['start'], unit['end'])
                if inds:
                    actual_visible.append([inds[0], inds[-1]+1])
                unit['in_graph'] = str(n['id']) in graph._edu_mapping
                units.append(unit)
            # Pinned preprocessing truncates the EDU prefix. Reject any internal holes.
            visible_nodes = [u for u in units if token_indices(offsets, u['start'], u['end'])]
            if ([u['node_id'] for u in visible_nodes] != [n['id'] for n in edu_nodes[:len(item['edu_token_spans'])]]
                    or actual_visible != [list(s) for s in item['edu_token_spans']]):
                raise ExplanationError('RST/token alignment is not a contiguous verified prefix')
            graph_json = graph_description(model, graph, item, units)
            root = graph._root_idx if graph._root_idx >= 0 else None
            method = 'race_contextual_edu_content_suppression'
        else:
            item = TextDataset([record], service.tokenizer)[0]
            units, root = display_segments(text), None
            method = 'roberta_word_embedding_suppression'
            if not hasattr(model.bert_model, 'embeddings') or not hasattr(model.bert_model.embeddings, 'word_embeddings'):
                raise ExplanationError('Unsupported encoder embedding interface')
        batch = collate_batch([item])
        if batch['full_text_input_ids'][0].tolist() != list(encoded['input_ids']):
            raise ExplanationError('Explanation tokenization differs from regular inference')
        prep.update(item=item, graph=graph, batch=batch, graph_json=graph_json, method=method)
        if baseline:
            _ensure_baseline(prep, model, race, graph, item)
    validate_units(units, text)
    if len(units) > 256:
        raise ExplanationError('Too many display units for interactive explanation (256)')
    for unit in units:
        indices = token_indices(offsets, unit['start'], unit['end'])
        unit['lexical_token_count'] = len(indices)
        unit['visible_end'] = min(unit['end'], covered)
        unit['visible_end_utf16'] = utf16_offset(text, max(unit['start'], min(unit['end'], covered)))
        if name == 'tfidf':
            # No position window: every unit inside the text is explainable.
            unit['coverage'] = 'full'
        else:
            unit['coverage'] = 'outside' if not indices else ('partial' if unit['end'] > covered and text[covered:unit['end']].strip() else 'full')
            if not indices:
                unit['status'] = 'outside_lexical_window'
    prep['units'] = units
    prep['units_kind'] = 'actual_rst_edus' if race else 'display_segments'
    return prep


def _preparation(service, key: str, text: str, name: str, document: dict, refresh: bool,
                 baseline: bool = True) -> tuple[dict, bool]:
    """Return (preparation, cache_hit). The caller holds service.lock."""
    if not refresh:
        with CACHE_LOCK:
            prep = PREP_CACHE.get(key)
            if prep is not None:
                PREP_CACHE.move_to_end(key)
                CACHE_STATS['hits'] += 1
                return prep, True
    prep = _build_preparation(service, text, name, document, baseline=baseline)
    with CACHE_LOCK:
        PREP_CACHE[key] = prep
        PREP_CACHE.move_to_end(key)
        while len(PREP_CACHE) > PREP_CACHE_ENTRIES:
            PREP_CACHE.popitem(last=False)
            CACHE_STATS['evictions'] += 1
        if refresh:
            CACHE_STATS['invalidations'] += 1
        else:
            CACHE_STATS['misses'] += 1
    return prep, False


def _validate_request(text, name, unit_ids, max_units) -> None:
    if name not in ('tfidf', 'roberta', 'race', 'race_no_supcon'):
        raise ExplanationError('Explanation is only implemented for the four main models')
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        raise ExplanationError('Enter non-empty English text, up to 12,000 characters')
    if isinstance(max_units, bool) or not isinstance(max_units, int) or not 1 <= max_units <= MAX_UNITS_PER_REQUEST:
        raise ExplanationError('max_units must be an integer from 1 to 8')
    if unit_ids is not None and (not isinstance(unit_ids, list) or not unit_ids or len(unit_ids) > max_units
                                 or any(not isinstance(x, str) for x in unit_ids) or len(set(unit_ids)) != len(unit_ids)):
        raise ExplanationError('unit_ids must be a nonempty unique string list within max_units')


def explain_document(service, text, name, unit_ids=None, max_units=8, refresh=False, prepare_only=False):
    """Serialize with the ORIGINAL service lock; never mutate weights or graph.

    The regular prediction remains available even if this optional adapter fails.
    max_seconds is cooperative between forwards, not hard cancellation.
    prepare_only returns the reusable metadata (units, graph, coverage) without
    running any intervention, so navigation works before influence is requested.
    """
    _validate_request(text, name, unit_ids, max_units)
    started = time.perf_counter()
    # Uses the unchanged original whole-document path; the parser runs once per
    # (text, model) and is reused by every later unit batch.
    document = _document_for(service, text, name)
    key = preparation_key(text, name, service)
    with service.lock:
        prep, cache_hit = _preparation(service, key, text, name, document, bool(refresh),
                                       baseline=not prepare_only)
        prepared_seconds = time.perf_counter() - started
        base_scores = prep['base_scores']
        model, offsets, covered = prep['model'], prep['offsets'], prep['covered']
        race, method = prep['race'], prep['method']
        units = copy.deepcopy(prep['units'])
        known = {u['id']: u for u in units}
        if unit_ids is not None and any(i not in known for i in unit_ids):
            raise ExplanationError('Unknown unit id')
        selected = [] if prepare_only else (
            unit_ids if unit_ids is not None else [u['id'] for u in units if u['coverage'] != 'outside'][:max_units])
        selected_set = set(selected)
        count = 0
        loop_started = time.perf_counter()
        stopped = False
        if selected_set and name != 'tfidf':
            # The traced baseline is deferred until influence is actually requested.
            _ensure_baseline(prep, model, race, prep['graph'], prep['item'])
        for unit in units:
            if unit['id'] not in selected_set or unit['coverage'] == 'outside':
                continue
            if count and time.perf_counter() - loop_started >= MAX_ANALYSIS_SECONDS:
                stopped = True
                break
            touched, trace = [], None
            if name == 'tfidf':
                modified = prep['x'].copy()
                share = prep['shares'][unit['id']]
                for k, j in enumerate(modified.indices):
                    modified.data[k] *= 1.0 - share.get(int(j), 0.0)
                altered = prep['classifier'].predict_proba(modified)[0]
                raw = np.asarray(prep['classifier'].decision_function(modified)[0], dtype=float)
                unit['affected_feature_count'] = len(share)
            else:
                hit = [0]
                if race:
                    def pre(_module, inputs, unit=unit):
                        hit[0] += 1
                        source = inputs[0]
                        if not torch.allclose(source, prep['base_trace'].unified, atol=5e-5, rtol=2e-5):
                            raise ExplanationError('Baseline node features changed during explanation')
                        changed, rows = race_suppressed_features(source, model.node_type_embedding.weight[0], prep['graph'],
                                                                prep['item']['nodes'], prep['item']['descendant_map'], unit['node_id'])
                        touched.extend(rows)
                        return (changed,) + inputs[1:]
                    context = temporary_hook(model.node_proj, pre, before=True)
                else:
                    indices = token_indices(offsets, unit['start'], unit['end'])
                    def emb_hook(_module, _inputs, output, indices=indices):
                        hit[0] += 1
                        out = output.clone()
                        out[:, indices, :] = 0
                        return out
                    context = temporary_hook(model.bert_model.embeddings.word_embeddings, emb_hook)
                with torch.inference_mode(), context, Trace(model, prep['graph']._root_idx if race and prep['graph']._root_idx >= 0 else None, race) as trace:
                    raw, altered = logits_and_scores(model(prep['batch']))
                if hit[0] != 1:
                    raise ExplanationError('Intervention boundary was not invoked exactly once')
            altered = score_array(altered)
            unit.update(status='computed', without_scores=scores_dict(altered),
                        delta_pp=dict(zip(LABELS, map(float, 100 * (base_scores - altered)))),
                        delta_logits=dict(zip(LABELS, map(float, prep['raw_base'] - raw))))
            if trace:
                unit['trace'] = trace.summaries(prep['base_trace'])
            if race:
                unit['directly_affected_node_indices'] = touched
            count += 1
        units_seconds = time.perf_counter() - loop_started
        # Re-run the ORIGINAL prediction path after all hooks are removed, without the parser.
        if prepare_only:
            restored_difference = None
        elif name == 'tfidf':
            restored_difference = check_parity(prep['classifier'].predict_proba(prep['x'])[0], base_scores,
                                               'Original prediction was not restored')
        else:
            with torch.inference_mode():
                _, final_probs = logits_and_scores(model(prep['batch']))
            restored_difference = check_parity(final_probs, base_scores, 'Original prediction was not restored')
        output = {'schema': SCHEMA, 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                  'document': document, 'method': method,
                  'units_kind': prep['units_kind'],
                  'units': units, 'graph': prep['graph_json'], 'stages': prep['stages'],
                  'checked_units': count, 'total_units': len(units), 'selected_unit_ids': selected,
                  'status': 'prepared' if prepare_only else ('budget_stopped' if stopped else 'complete_for_requested_units'),
                  'baseline_replay_max_abs_diff': (None if prep['replay'] is None
                                                   else float(np.max(np.abs(np.asarray(prep['replay'])-base_scores)))),
                  'restored_max_abs_diff': restored_difference,
                  'elapsed_seconds': time.perf_counter() - started,
                  'timing': {'prepare_seconds': prepared_seconds, 'units_seconds': units_seconds},
                  'cache': {'preparation': 'hit' if cache_hit else 'miss', **cache_stats()},
                  'definition': 'delta_pp = 100 × (original document score − intervened document score)',
                  'reference': 'zero content at the explicitly named internal boundary',
                  'limitations': [
                      'Sensitivity to a zero-feature reference, not a verified authorship label.',
                      'Effects can interact and do not sum to the document score or 100%.',
                      'Different model intervention boundaries are not comparable as importance rankings.',
                      'RACE retains other contextual vectors, node types, graph topology and relations.',
                      'Layer RMS changes show activation changes, not class contributions or edge weights.',
                      'Uncomputed or out-of-window units have no influence estimate, not zero influence.'
                  ]}
        return output
