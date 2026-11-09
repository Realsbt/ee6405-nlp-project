"""Design integrity, interaction arithmetic and validation-only selection gates."""
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.joint_research import (base_config, candidates, choose, execute_joint,
    factorial_effects, finalize_comparison, paired_validation)
from experiments.run_research import make_run_id, main, Pipeline


def result(config, name, f1=.8, auc=.98, minority=.75):
    return {'config': copy.deepcopy(config), 'run_id': name, 'checkpoint_sha256': 'synthetic',
            'validation': {'macro_f1': f1, 'macro_auroc_ovr': auc,
                           'per_class': {'ai_humanized': {'f1': minority}}}}


class JointResearchTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent/'configs/a4000_joint_v2.json').read_text())
        self.base = base_config(self.config); self.spec = self.config['joint_plan']

    def test_decimal_run_names_and_length_limits(self):
        name = make_run_id('lr1e-05_decay0.8', 3407, '2026-11-09_09-00-00')
        self.assertNotIn('.', name); self.assertIn('decay0p8', name)
        from argparse import ArgumentTypeError
        with self.assertRaises(ArgumentTypeError): make_run_id('x'*80, 3407)
        for item in candidates(self.config): make_run_id(item['tag'], 3407)

    def test_complete_unique_factorial_and_two_separate_controls(self):
        rows = candidates(self.config)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r['tag'] for r in rows}), 10)
        self.assertEqual({tuple(r['bits']) for r in rows if r['bits'] is not None}, set(itertools.product((0,1), repeat=3)))
        self.assertEqual(rows[0]['config'], self.base)
        self.assertTrue(all(r['config']['batch_size'] == 16 and r['config']['architecture']['unfreeze_bert_layers'] == 6 for r in rows))
        bad = copy.deepcopy(self.config); bad['joint_plan']['confirmation_seeds'] = [42]
        with self.assertRaises(ValueError): candidates(bad)

    def test_selection_rejects_ranking_or_minority_regressions(self):
        anchor = result(self.base, 'anchor')
        auc_bad = result(self.base, 'bad_auc', f1=.9, auc=.97)
        minority_bad = result(self.base, 'bad_minority', f1=.9, minority=.7)
        good = result(self.base, 'good', f1=.82)
        auc_bad['test'] = {'macro_f1': 1.0}
        self.assertIs(choose([auc_bad, minority_bad], anchor, self.spec), anchor)
        self.assertIs(choose([auc_bad, good], anchor, self.spec), good)

    def test_paired_gate_rejects_one_lucky_seed_and_missing_seeds(self):
        old, new = [], []
        for seed, delta in zip((3407,42,2026), (.04,-.005,-.005)):
            c = dict(self.base, seed=seed)
            old.append(result(c, f'b{seed}'))
            new.append(result(c, f'n{seed}', f1=.8+delta))
        self.assertFalse(paired_validation(old, new, self.spec)['passed'])
        with self.assertRaises(ValueError): paired_validation(old, new[:2], self.spec)
        for r in new: r['validation']['macro_f1'] = .82
        self.assertTrue(paired_validation(old, new, self.spec)['passed'])

    def test_known_main_pair_and_triple_interactions(self):
        cells = {}
        for p,r,h in itertools.product((0,1), repeat=3):
            cells[p,r,h] = result(self.base, 'synthetic', f1=.8+.01*p+.02*r+.03*h+.04*p*r+.05*p*r*h)
        e = factorial_effects(cells)['effects']['macro_f1']
        self.assertAlmostEqual(e['mean_main_effect_pp']['projection'], 4.25)
        self.assertAlmostEqual(e['mean_pair_interaction_pp']['projection__weak_supcon'], 6.5)
        self.assertAlmostEqual(e['triple_interaction_pp'], 5.0)
        self.assertAlmostEqual(e['joint_gain_minus_sum_single_gains_pp'], 9.0)
        del cells[0,0,0]
        with self.assertRaises(ValueError): factorial_effects(cells)

    def simulate_pipeline(self, improve):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)/'batch'; batch.mkdir()
            old = Path(directory)/'anchor'; old.mkdir()
            anchor = result(self.base, 'existing_anchor')
            pipeline = SimpleNamespace(batch=batch, state={'run_root': str(batch/'runs'), 'completed_runs': []},
                                       save=Mock(), operation=Mock(), command=Mock(), finalize=Mock())
            def train(tag, config):
                value = .82 if improve and (tag == 'f_p1_r1_h1' or tag == 'joint_selected') else .8
                return result(config, f'{tag}_s{config["seed"]}', f1=value)
            pipeline.train = Mock(side_effect=train)
            with patch('experiments.joint_research.verify_anchor', return_value=anchor), \
                 patch('experiments.joint_research.anchor_path', return_value=old), \
                 patch('experiments.joint_research.freeze_reference', return_value=batch/'reference.json') as freeze:
                execute_joint(pipeline, self.config)
            confirmation = json.loads((batch/'joint_validation.json').read_text())
            self.assertEqual(pipeline.train.call_count, 13 if improve else 11)
            self.assertEqual(pipeline.command.call_count, 0)
            self.assertEqual(freeze.call_count, 0)
            self.assertEqual(confirmation['used_anchor_fallback'], not improve)
            self.assertFalse(confirmation['selection_uses_test'])
            members = pipeline.finalize.call_args.args[0]
            self.assertEqual([m['config']['seed'] for m in members], [3407,42,2026])
            if improve: self.assertTrue(confirmation['passed'])
            else: self.assertIs(members[0], anchor)
            self.assertEqual(pipeline.finalize.call_args.kwargs['baseline_members'] is not None, improve)

    def test_promising_combination_trains_exactly_thirteen_new_runs(self):
        self.simulate_pipeline(True)

    def test_failed_screening_retains_anchor_without_candidate_test(self):
        self.simulate_pipeline(False)

    def test_missing_gpu_never_constructs_pipeline_or_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'batch'
            with patch('sys.argv', ['run', '--name', 'joint_test', '--output', str(out), '--joint']), \
                 patch('experiments.joint_research.preflight_plan'), \
                 patch('torch.cuda.is_available', return_value=False), \
                 patch('experiments.run_research.Pipeline') as pipeline, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught: main()
                self.assertEqual(caught.exception.code, 2)
                pipeline.assert_not_called(); self.assertFalse(out.exists())

    def test_both_manifests_are_frozen_before_first_test_command(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); batch = folder/'batch'; batch.mkdir()
            root = batch/'runs'; root.mkdir(); data = folder/'data'; data.mkdir()
            (data/'protocol.json').write_text('{}')
            groups = []
            for kind in ('base', 'candidate'):
                rows = []
                for seed in (3407,42,2026):
                    name = f'{kind}_s{seed}'; run = root/name; run.mkdir()
                    (run/'best.pt').write_bytes(b'weights'); (run/'config.json').write_text('{}')
                    rows.append({'run_id': name, 'code_sha256': {}})
                groups.append(rows)
            def first_command(*args, **kwargs):
                for path in (batch/'final_manifest.json', batch/'baseline_confirmation/final_manifest.json'):
                    self.assertTrue(path.exists()); self.assertTrue(path.with_suffix('.sha256').exists())
                    self.assertEqual(len(json.loads(path.read_text())['members']), 3)
                raise RuntimeError('stop before any inference')
            pipeline = SimpleNamespace(batch=batch, state={'run_root': str(root), 'git_commit': 'test', 'initial_commit': 'test'},
                                       args=SimpleNamespace(single_config=False), command=Mock(side_effect=first_command))
            with patch('experiments.run_research.RUN_ROOT', root), patch('experiments.run_research.DATA_ROOT', data), \
                 patch('experiments.joint_research.DATA_ROOT', data):
                with self.assertRaisesRegex(RuntimeError, 'stop before any inference'):
                    Pipeline.finalize(pipeline, groups[1], baseline_members=groups[0])

    def test_publication_requires_gain_over_six_layer_reference_and_auc_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory); (batch/'final_evaluation').mkdir()
            (batch/'baseline_confirmation/final_evaluation').mkdir(parents=True)
            def final(f1, auc):
                return {'members': [{'seed': s, 'test': {'macro_f1': f1, 'macro_auroc_ovr': auc}} for s in (3407,42,2026)]}
            (batch/'final_evaluation/result.json').write_text(json.dumps(final(.93,.97)))
            (batch/'baseline_confirmation/final_evaluation/result.json').write_text(json.dumps(final(.92,.98)))
            (batch/'pipeline.json').write_text(json.dumps({'joint_anchor_selected': False, 'joint_auroc_slack': .001}))
            (batch/'score_change.json').write_text(json.dumps({'publish_eligible': True}))
            finalize_comparison(batch)
            self.assertFalse(json.loads((batch/'score_change.json').read_text())['publish_eligible'])
            self.assertAlmostEqual(json.loads((batch/'joint_comparison.json').read_text())['f1_gain_pp'], 1.0)


if __name__ == '__main__': unittest.main()
