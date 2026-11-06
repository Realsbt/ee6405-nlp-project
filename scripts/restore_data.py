"""Reconstruct exact frozen HART/HC3 text partitions from separately obtained originals."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paths import ASSETS, STORAGE, DATA_ROOT
from experiments.common import file_hash, write_jsonl
from experiments.neural import tokenizer_local
from experiments.prepare_data import load_release, clean_and_group, make_partitions, text_hash


def hart():
    protocol_path = ROOT / 'experiments/reports/data_v1.json'
    protocol = json.loads(protocol_path.read_text())
    if DATA_ROOT.exists():
        for name in ('train', 'val', 'test'):
            assert file_hash(DATA_ROOT / f'{name}.jsonl') == protocol['partitions'][name]['sha256']
        assert file_hash(DATA_ROOT / 'protocol.json') == file_hash(protocol_path)
        print('Existing HART partitions match; preserved.')
        return
    recorded = {}
    for row in protocol['source_files']:
        path = ASSETS / 'hart_official' / row['name']
        assert file_hash(path) == row['sha256'], path
        recorded[row['name']] = row['path']
    raw = load_release(ASSETS / 'hart_official')
    for row in raw:
        row['source_file'] = recorded[Path(row['source_file']).name]
    retained, exclusions = clean_and_group(raw)
    partitions = make_partitions(retained)
    tokenizer = tokenizer_local()
    for rows in partitions.values():
        for start in range(0, len(rows), 128):
            chunk = rows[start:start+128]
            tokens = tokenizer([r['article'] for r in chunk], truncation=True, max_length=512,
                               return_offsets_mapping=True, return_attention_mask=False)
            for row, offsets, ids in zip(chunk, tokens['offset_mapping'], tokens['input_ids']):
                end = max(end for begin, end in offsets if end > begin)
                row['model_text'] = row['article'][:end]
                row['input_token_count_with_specials'] = len(ids)
                row['input_truncated'] = end < len(row['article'].rstrip())
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DATA_ROOT.parent, prefix='restore-hart-') as temporary:
        output = Path(temporary) / 'data'
        output.mkdir()
        for name, rows in partitions.items():
            path = output / f'{name}.jsonl'
            write_jsonl(path, rows)
            assert file_hash(path) == protocol['partitions'][name]['sha256'], f'{name}: reconstructed bytes differ'
        write_jsonl(output / 'exclusions.jsonl', exclusions)
        shutil.copyfile(protocol_path, output / 'protocol.json')
        output.rename(DATA_ROOT)
    print('HART restored: exact original train/validation/test hashes, 11,190 / 1,600 / 3,194 records.')


def hc3():
    recipe = json.loads((ROOT / 'resources/hc3_selection.json').read_text())
    protocol = recipe['protocol']
    target = STORAGE / 'experiments/hc3_aux_v1_1'
    if target.exists():
        assert file_hash(target / 'test.jsonl') == protocol['answer_sha256']
        assert json.loads((target / 'protocol.json').read_text()) == protocol
        print('Existing HC3 partition matches; preserved.')
        return
    raw = ASSETS / 'hc3/all.jsonl'
    expected = next(r['sha256'] for r in protocol['source']['files'] if r['name'] == 'all.jsonl')
    assert file_hash(raw) == expected
    wanted = {row['metadata']['normalized_text_sha256'] for row in recipe['rows']}
    answers = {}
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            for key in ('human_answers', 'chatgpt_answers'):
                answer = next((a.strip() for a in row[key] if isinstance(a, str) and a.strip()), None)
                if answer is not None:
                    fingerprint = text_hash(answer)
                    if fingerprint in wanted:
                        answers.setdefault(fingerprint, answer)
    records = []
    for entry in recipe['rows']:
        values = dict(entry['metadata'])
        values['article'] = answers[values['normalized_text_sha256']]
        values['model_text'] = values['article'][:entry['model_text_characters']]
        records.append({key: values[key] for key in entry['fields']})
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='restore-hc3-') as temporary:
        output = Path(temporary) / 'data'
        output.mkdir()
        write_jsonl(output / 'test.jsonl', records)
        assert file_hash(output / 'test.jsonl') == protocol['answer_sha256'], 'HC3 reconstructed bytes differ'
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
        output.rename(target)
    print('HC3 restored: 1,000 records with the original frozen answer-file hash.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['hart', 'hc3', 'all'], default='all')
    args = parser.parse_args()
    if args.dataset in ('hart', 'all'):
        hart()
    if args.dataset in ('hc3', 'all'):
        hc3()


if __name__ == '__main__':
    main()
