"""Stage B: controlled test of surface-cue-aware training reweighting.

Fair comparison, as required:
  * both arms start from the SAME checkpoint (roberta_s3407 best.pt)
  * both arms run the SAME number of extra epochs
  * both arms keep the same optimiser, schedule, batch sampler and seed
  * the ONLY difference is the per-sample weight in the cross-entropy loss

Arm A (control): plain continued training, uniform weights.
Arm B (treatment): weights w_i = 1 + (1 - b_i) from the group-disjoint cue
model, normalised per class.

Selection is on validation macro-F1, identical for both arms. The test set is
scored once at the end for both arms.

Writes `.local/reweight_experiment.json`.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.neural import (  # noqa: E402
    BalancedBatchSampler, TextDataset, collate_batch, load_records, tokenizer_local,
)
from experiments.paths import resolve_recorded_path  # noqa: E402
from experiments.train_neural import build_model, evaluate, import_upstream  # noqa: E402

OUT = ROOT / '.local/reweight_experiment.json'
LABELS = ['human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized']
EXTRA_EPOCHS = 3
SEED = 3407
BASE_RUN = 'roberta_s3407'


def find_base_checkpoint() -> pathlib.Path:
    candidates = [
        pathlib.Path.home() / f'ee6405-resources/experiments/neural_runs/{BASE_RUN}/best.pt',
        pathlib.Path('E:/ee6405-ckpt') / f'{BASE_RUN}-best.pt',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'no checkpoint for {BASE_RUN}')


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt_path = find_base_checkpoint()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = dict(ckpt['config'])
    config['architecture'] = dict(config['architecture'])
    config['architecture']['backbone_model_path'] = str(
        resolve_recorded_path(config['architecture']['backbone_model_path']))
    print(f'base checkpoint {ckpt_path}', flush=True)
    print(f'  best_epoch {ckpt.get("best_epoch")}  lr {config["learning_rate"]}  '
          f'bs {config["batch_size"]}', flush=True)

    tokenizer = tokenizer_local()
    train_records = load_records('train', pathlib.Path('experiments/hart_grouped_v1'))
    val_records = load_records('val', pathlib.Path('experiments/hart_grouped_v1'))
    train_data = TextDataset(train_records, tokenizer)
    val_data = TextDataset(val_records, tokenizer)
    print(f'  train {len(train_data)}  val {len(val_data)}', flush=True)

    # ---- weights ---------------------------------------------------------
    cw = json.loads((ROOT / '.local/cue_weights.json').read_text(encoding='utf-8'))
    wmap = {it['item_id']: it['weight'] for it in cw['items']}
    weights = np.asarray([wmap.get(r['item_id'], 1.0) for r in train_records], dtype=np.float64)
    print(f'  weights mean {weights.mean():.3f}  min {weights.min():.3f}  '
          f'max {weights.max():.3f}', flush=True)

    results = {}
    for arm, use_weights in (('control_uniform', False), ('treatment_cue_weighted', True)):
        print(f'\n=== {arm} ===', flush=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = build_model(config, None).cuda()
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for _, p in trainable],
                                      lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])
        sampler = BalancedBatchSampler(train_data.labels, config['batch_size'], SEED)
        loader = DataLoader(train_data, batch_sampler=sampler,
                            collate_fn=collate_batch, num_workers=0)
        total_steps = EXTRA_EPOCHS * len(loader)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * config['warmup_ratio']), total_steps)

        best_score, best_state, best_epoch = -1.0, None, -1
        started = time.perf_counter()
        for epoch in range(EXTRA_EPOCHS):
            model.train()
            sampler.epoch = 1000 + epoch
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(batch)
                    labels = batch['labels'].cuda()
                    logits = prediction['logits'].float()
                    if use_weights:
                        w = torch.tensor(
                            [wmap.get(i, 1.0) for i in batch['id']],
                            dtype=torch.float32, device=logits.device)
                        per_item = F.cross_entropy(logits, labels, reduction='none')
                        loss = (per_item * w).sum() / w.sum()
                    else:
                        loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable],
                                               config['gradient_clip'],
                                               error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
            metrics, _, _, _ = evaluate(model, val_data, config['batch_size'])
            print(f'  epoch {epoch+1}: val acc {100*metrics["accuracy"]:.2f}%  '
                  f'f1 {100*metrics["macro_f1"]:.2f}%', flush=True)
            if metrics['macro_f1'] > best_score:
                best_score = metrics['macro_f1']
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elapsed = time.perf_counter() - started
        model.load_state_dict(best_state, strict=True)
        results[arm] = {
            'val_macro_f1': float(best_score),
            'best_epoch': best_epoch,
            'seconds': elapsed,
            'state': best_state,
        }
        print(f'  best val f1 {100*best_score:.2f}% at epoch {best_epoch} '
              f'({elapsed:.0f}s)', flush=True)

    # ---- score test once per arm -----------------------------------------
    print('\n=== test evaluation ===', flush=True)
    test_records = load_records('test', pathlib.Path('experiments/hart_grouped_v1'))
    test_data = TextDataset(test_records, tokenizer)
    truth = np.asarray([LABELS.index(r['label']) for r in test_records])
    payload = {'protocol': {
        'base_checkpoint': str(ckpt_path),
        'extra_epochs': EXTRA_EPOCHS,
        'same_start': True, 'same_updates': True, 'same_seed': True,
        'only_difference': 'per-sample weight in cross-entropy',
        'selection': 'validation macro-F1, identical for both arms',
    }, 'arms': {}}
    for arm, entry in results.items():
        model = build_model(config, None).cuda()
        model.load_state_dict(entry['state'], strict=True)
        metrics, probs, ids, _ = evaluate(model, test_data, config['batch_size'])
        payload['arms'][arm] = {
            'val_macro_f1': entry['val_macro_f1'],
            'best_epoch': entry['best_epoch'],
            'seconds': entry['seconds'],
            'test_accuracy': float(metrics['accuracy']),
            'test_macro_f1': float(metrics['macro_f1']),
            'probabilities': probs.tolist(),
            'item_ids': list(ids),
        }
        print(f'  {arm:26s} test {100*metrics["accuracy"]:.2f}% / '
              f'{100*metrics["macro_f1"]:.2f}%', flush=True)

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
