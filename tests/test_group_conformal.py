import copy
import itertools
import unittest

from experiments.group_conformal import LABELS, assert_disjoint, fit, predict_set, run, split_groups, validate_rows


def row(index, group, true_probability=0.7):
    return {"item_id": str(index), "group_id": str(group), "domain": "example",
            "true_label": LABELS[0], "probabilities": dict(zip(LABELS, [true_probability] + [(1-true_probability)/3]*3))}


class GroupConformalTests(unittest.TestCase):
    def test_group_max_and_finite_sample_rank(self):
        rows = [row(i, i, 1-i/10) for i in range(1, 10)]
        rows.append(row(99, 1, 0.05))
        artifact = fit(rows, 0.2)
        self.assertEqual(artifact["calibration_groups"], 9)
        self.assertEqual(artifact["quantile_rank"], 8)
        self.assertAlmostEqual(artifact["threshold"], 0.9)

    def test_small_calibration_includes_all_labels(self):
        artifact = fit([row(0, 0)], 0.1)
        result = predict_set(row(1, 1)["probabilities"], artifact)
        self.assertEqual(result["prediction_set"], list(LABELS))
        self.assertTrue(result["needs_review"])

    def test_empty_set_is_reviewed_without_argmax_fallback(self):
        artifact = fit([row(i, i, 1.0) for i in range(10)], 0.1)
        result = predict_set(dict.fromkeys(LABELS, 0.25), artifact)
        self.assertEqual(result["prediction_set"], [])
        self.assertEqual(result["review_reason"], "empty_set")

    def test_source_isolation_and_order_independence(self):
        rows = [row(i, i//4) for i in range(80)]
        a, b = split_groups(rows)
        assert_disjoint(a, b)
        reversed_a, _ = split_groups(list(reversed(rows)))
        self.assertEqual({r["item_id"] for r in a}, {r["item_id"] for r in reversed_a})
        changed = copy.deepcopy(rows)
        for r in changed:
            r["true_label"] = LABELS[2]
        self.assertEqual({r["item_id"] for r in a}, {r["item_id"] for r in split_groups(changed)[0]})
        with self.assertRaises(ValueError):
            run(a, a, 0.1)

    def test_invalid_and_duplicate_predictions(self):
        for p in (float("nan"), float("inf"), -0.1, 1.1):
            r = row(1, 1)
            r["probabilities"][LABELS[0]] = p
            with self.assertRaises(ValueError):
                validate_rows([r])
        with self.assertRaises(ValueError):
            validate_rows([row(1, 1), row(1, 2)])

    def test_evaluation_labels_do_not_change_calibrator(self):
        a = [row(i, i) for i in range(20)]
        b = [row(100+i, 100+i) for i in range(5)]
        first = run(a, b, 0.1)[0]
        for r in b:
            r["true_label"] = LABELS[1]
        self.assertEqual(first, run(a, b, 0.1)[0])

    def test_exact_exchangeable_group_rank_coverage(self):
        # Exhaust every ordering of five distinct group scores. With four
        # calibration groups and alpha=.2, the fifth rank fails exactly 1/5.
        hits = total = 0
        for order in itertools.permutations(range(5)):
            calibration = [row(i, i, 1-(score+1)/10) for i, score in enumerate(order[:4])]
            artifact = fit(calibration, 0.2)
            candidate = row(9, 9, 1-(order[4]+1)/10)
            hits += LABELS[0] in predict_set(candidate["probabilities"], artifact)["prediction_set"]
            total += 1
        self.assertAlmostEqual(hits/total, 0.8)


if __name__ == "__main__":
    unittest.main()
