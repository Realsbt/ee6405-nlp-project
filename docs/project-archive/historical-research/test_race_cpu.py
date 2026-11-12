import unittest
import numpy as np
from race_cpu_replay import fit_temperature, temperature_probabilities, compare
from calibrate_saved_models import probability_logits
from replay_hc3_cpu import LABELS


class CalibrationTests(unittest.TestCase):
    def test_identity_and_offset_invariance(self):
        logits = np.array([[1., 2, 3, 4], [4., 0, -2, 1]])
        np.testing.assert_allclose(temperature_probabilities(logits, 1),
                                   temperature_probabilities(logits + 50, 1))
        np.testing.assert_allclose(temperature_probabilities(logits, 2).sum(1), 1)
        np.testing.assert_array_equal(temperature_probabilities(logits, 2).argmax(1), logits.argmax(1))

    def test_fit_does_not_increase_validation_nll(self):
        result = fit_temperature([[8, 0, 0, 0]] * 4, [0, 1, 2, 3])
        self.assertGreater(result["temperature"], 1)
        self.assertLessEqual(result["validation_nll_after"], result["validation_nll_before"])

    def test_invalid_input(self):
        for temp in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                temperature_probabilities([[1, 2, 3, 4]], temp)
        with self.assertRaises(ValueError):
            fit_temperature([[1, 2, 3, 4]], [5])

    def test_stored_probabilities_and_replay_comparison(self):
        p = [0.1, 0.2, 0.3, 0.4]
        records = [{"probabilities": dict(zip(LABELS, p))}]
        z = probability_logits(records)
        np.testing.assert_allclose(temperature_probabilities(z, 1), [p])
        self.assertEqual(compare(z, records, "probabilities")["status"], "passed")
        self.assertEqual(compare(-z, records, "probabilities")["status"], "review_required")
        records[0]["probabilities"][LABELS[0]] = 0
        with self.assertRaises(ValueError):
            probability_logits(records)


if __name__ == "__main__":
    unittest.main()
