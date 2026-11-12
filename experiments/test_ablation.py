"""Regression checks for independent architecture/loss controls and run identity."""

import json
from pathlib import Path
import unittest

from experiments.ablation import CONDITIONS, MISSING_RUNS, NEURAL_RUNS, condition_name, recipe_for_run, resolve_config
from experiments.factorial_summary import summarize_factorial


class AblationChecks(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((Path(__file__).parent / "configs/neural_v1.json").read_text())

    def test_json_supcon_survives_architecture_override(self):
        self.base["use_supcon"] = True
        for architecture in ("roberta", "race"):
            config = resolve_config(self.base, architecture)
            self.assertTrue(config["use_supcon"])
        self.assertEqual(condition_name(resolve_config(self.base, "roberta")), "roberta_supcon")

    def test_cli_overrides_each_factor_independently(self):
        for kind, (architecture, supcon) in CONDITIONS.items():
            config = resolve_config(self.base, architecture, 42, supcon)
            self.assertEqual(condition_name(config), kind)
            self.assertEqual(config["seed"], 42)
            for key in set(self.base) - {"model_type", "seed", "use_supcon"}:
                self.assertEqual(config[key], self.base[key])
        self.assertFalse(self.base["use_supcon"])

    def test_legacy_ablation_alias_and_contradiction(self):
        legacy = dict(self.base, model_type="race_no_supcon", use_supcon=False)
        self.assertEqual(resolve_config(legacy), recipe_for_run(self.base, "race_no_supcon_s3407"))
        with self.assertRaises(ValueError):
            resolve_config(self.base, "race_no_supcon", use_supcon=True)

    def test_all_cells_have_same_seeds_and_unique_names(self):
        self.assertEqual(len(set(NEURAL_RUNS)), 12)
        self.assertEqual(len(MISSING_RUNS), 5)
        for run in NEURAL_RUNS:
            config = recipe_for_run(self.base, run)
            self.assertEqual(f"{condition_name(config)}_s{config['seed']}", run)

    def test_invalid_loss_is_rejected(self):
        for value in ("false", 1, None):
            with self.assertRaises(ValueError):
                resolve_config(dict(self.base, use_supcon=value))

    def test_paired_effects_and_interaction_sign(self):
        # Known pattern: SupCon helps RoBERTa by .01 and RACE by .03.
        analyses = {f"{kind}_s{seed}": {"test": {metric: value + offset for metric in
                    ("accuracy", "macro_f1", "macro_auroc_ovr")}}
                    for kind, value in (("roberta", .80), ("roberta_supcon", .81),
                                        ("race_no_supcon", .82), ("race", .85))
                    for seed, offset in ((3407, 0), (42, .01), (2026, -.01))}
        report = summarize_factorial(analyses)
        self.assertEqual(report["status"], "completed")
        for name, expected in (("supcon_with_roberta", .01), ("supcon_with_race", .03),
                               ("race_minus_roberta_with_ce", .02),
                               ("race_minus_roberta_with_supcon", .04), ("interaction", .02)):
            self.assertAlmostEqual(report["effects"][name]["metrics"]["macro_f1"]["mean"], expected)
        del analyses["roberta_supcon_s42"]
        partial = summarize_factorial(analyses)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["effects"]["interaction"]["seeds"], [3407, 2026])
        self.assertEqual(partial["effects"]["supcon_with_race"]["seeds"], [3407, 42, 2026])


if __name__ == "__main__":
    unittest.main()
