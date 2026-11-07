import unittest

import numpy as np

from experiments.evaluate_multiview import probability_mean, windows_for_ids


class MultiviewTests(unittest.TestCase):
    def test_short_document_has_one_window(self):
        self.assertEqual(windows_for_ids(list(range(10))), [list(range(10))])

    def test_long_document_uses_head_middle_tail(self):
        ids = list(range(700))
        windows = windows_for_ids(ids, 512)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], ids[:512])
        self.assertEqual(windows[2], ids[-512:])
        self.assertEqual(windows[1], ids[94:606])

    def test_mean_is_over_probabilities(self):
        values = [np.array([.8, .1, .05, .05]), np.array([.2, .3, .4, .1])]
        np.testing.assert_allclose(probability_mean(values), [.5, .2, .225, .075])

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            probability_mean([np.array([.5, .5, 0, .1])])


if __name__ == "__main__":
    unittest.main()
