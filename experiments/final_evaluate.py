"""Evaluate only members frozen in a hash-checked final manifest."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.common import DATA_ROOT, LABELS, DOMAINS, classification_metrics, file_hash, write_json, write_jsonl
from experiments.neural import TextDataset, build_model, load_records, tokenizer_local
from experiments.train_neural import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = p.parse_args()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    manifest = json.loads(args.manifest.read_text())
    expected = args.manifest.with_suffix('.sha256').read_text().strip()
    if file_hash(args.manifest) != expected:
        raise ValueError('Frozen manifest changed')
    for source, digest in manifest['evaluation_source_sha256'].items():
        if file_hash(Path(source)) != digest:
            raise ValueError('Evaluation source changed')
    output = args.manifest.parent / 'final_evaluation'
    if output.exists():
        raise FileExistsError('Preserving previous final evaluation')
    if file_hash(DATA_ROOT / 'protocol.json') != manifest['data_protocol_sha256']:
        raise ValueError('Data protocol changed')
    # Validate all checkpoints/configs before opening test.
    for member in manifest['members']:
        for path_key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('config', 'config_sha256')):
            if file_hash(Path(member[path_key])) != member[hash_key]:
                raise ValueError('Frozen member changed')
        for path, digest in member['source_sha256'].items():
            if file_hash(Path(path)) != digest:
                raise ValueError('Frozen training source changed')
    weights = np.asarray([m['weight'] for m in manifest['members']])
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid fixed ensemble weights')
    output.mkdir()
    records = load_records('test')
    data = TextDataset(records, tokenizer_local())
    results, ensemble = [], None
    torch.set_num_threads(4)
    for member in manifest['members']:
        config = json.loads(Path(member['config']).read_text())
        print(f"Evaluating {member['run_id']} on {device} in float32", flush=True)
        model = build_model(config).to(device)
        ckpt = torch.load(member['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        metrics, probs, ids, seconds = evaluate(model, data, config['batch_size'], progress_interval=10)
        assert ids == [r['item_id'] for r in records]
        domain = {}
        for d in DOMAINS:
            mask = np.asarray([r['domain'] == d for r in records])
            domain[d] = classification_metrics([r['label_id'] for r in records if r['domain'] == d], probs[mask])
        row = {'run_id': member['run_id'], 'seed': config['seed'], 'test': metrics, 'test_by_domain': domain,
               'seconds': seconds, 'device': str(device), 'precision': 'float32', 'checkpoint_sha256': member['checkpoint_sha256']}
        write_json(output / f"{member['run_id']}.json", row)
        write_jsonl(output / f"{member['run_id']}_predictions.jsonl", [
            {'item_id': r['item_id'], 'true_label': r['label'], 'probabilities': dict(zip(LABELS, v.tolist()))}
            for r, v in zip(records, probs)])
        results.append(row)
        ensemble = probs * member['weight'] if ensemble is None else ensemble + probs * member['weight']
        del model, ckpt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(output / 'result.json', {'status': 'completed', 'manifest_sha256': file_hash(args.manifest),
               'members': results, 'ensemble': classification_metrics([r['label_id'] for r in records], ensemble),
               'test_partition_loaded': True})
    write_jsonl(output / 'ensemble_predictions.jsonl', [
        {'item_id': r['item_id'], 'true_label': r['label'], 'probabilities': dict(zip(LABELS, v.tolist()))}
        for r, v in zip(records, ensemble)])


if __name__ == '__main__':
    main()
