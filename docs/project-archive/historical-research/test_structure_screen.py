import copy
import unittest

import numpy as np

from screen_structure_cpu import LABELS, align_predictions, features


class StructureScreenTests(unittest.TestCase):
    def record(self):
        return {'article': 'One. Two words.', 'label': 'human_written',
                'domain': 'writing', 'model_name': 'private-generator',
                'rst_structure': {'start': 0, 'end': 15, 'relation': 'Joint', 'nuclearity': 'NN',
                    'left': {'start': 0, 'end': 4, 'text': 'One.', 'left': None, 'right': None},
                    'right': {'start': 5, 'end': 15, 'text': 'Two words.', 'left': None, 'right': None}}}

    def test_known_tree(self):
        result = features(self.record())
        self.assertEqual(result['structure']['leaf_depth_max'], 1)
        self.assertEqual(result['structure']['edu_words_mean'], 1.5)
        self.assertEqual(result['structure']['relation:Joint'], 1)
        self.assertAlmostEqual(result['structure']['span_imbalance_mean'], 6/14)
        self.assertTrue(all(np.isfinite(v) for group in result.values() for v in group.values()))

    def test_metadata_cannot_affect_features(self):
        original = self.record()
        changed = copy.deepcopy(original)
        changed.update(label='ai_generated', domain='arxiv', model_name='another', item_id='secret')
        self.assertEqual(features(original), features(changed))

    def test_predictions_align_by_id_and_reject_wrong_labels(self):
        records = [{'item_id': i, 'group_id': i, 'label': LABELS[n]} for n, i in enumerate(('a', 'b'))]
        predictions = [{'item_id': r['item_id'], 'group_id': r['group_id'],
                        'true_label': r['label'], 'probabilities': dict(zip(LABELS, p))}
                       for r, p in zip(records, ([.7, .1, .1, .1], [.1, .7, .1, .1]))]
        result = align_predictions(records, predictions[::-1])
        np.testing.assert_array_equal(result.argmax(1), [0, 1])
        predictions[0]['true_label'] = LABELS[2]
        with self.assertRaises(ValueError):
            align_predictions(records, predictions)

    def test_duplicate_predictions_rejected(self):
        with self.assertRaises(ValueError):
            align_predictions([{'item_id': 'a'}], [{'item_id': 'a'}, {'item_id': 'a'}])


if __name__ == '__main__':
    unittest.main()
