import unittest
import numpy as np
from sklearn.metrics import f1_score
from improve_cpu import specialize, bias_humanized, score, bootstrap_deltas, validate_probabilities


class ImprovementTests(unittest.TestCase):
    def test_specialist_preserves_creator_mass(self):
        p = np.array([[.1,.2,.3,.4],[.7,.1,.15,.05]])
        q = np.array([[.9,.1],[.2,.8]])
        for beta in (0,.25,1):
            r = specialize(p,q,beta)
            np.testing.assert_allclose(r[:,:2],p[:,:2])
            np.testing.assert_allclose(r[:,2:].sum(1),p[:,2:].sum(1))
            np.testing.assert_allclose(r.sum(1),1)
        np.testing.assert_allclose(specialize(p,q,0),p)

    def test_reject_bad_probabilities(self):
        with self.assertRaises(ValueError):
            validate_probabilities([[1,0,0,-.1]])
        with self.assertRaises(ValueError):
            specialize(np.ones((1,4))/4,np.array([[.7,.7]]),1)

    def test_bias_identity_and_mass(self):
        p=np.array([[.1,.2,.3,.4]])
        np.testing.assert_allclose(bias_humanized(p,1),p)
        r=bias_humanized(p,2)
        np.testing.assert_allclose(r.sum(1),1)
        self.assertGreater(r[0,3],p[0,3])

    def test_score_matches_sklearn(self):
        y=np.array([0,0,1,2,3])
        p=np.eye(4)[[0,1,1,2,2]]
        self.assertAlmostEqual(score(y,p),f1_score(y,p.argmax(1),labels=range(4),average='macro'))

    def test_paired_bootstrap_identical_is_zero(self):
        r=[{'group_id':str(i//2)} for i in range(8)]
        y=np.array([0,1,2,3]*2)
        p=np.eye(4)[y]
        result=bootstrap_deltas(r,y,{'a':p},{'b':p},n=30)['a']['b']
        self.assertEqual(result['delta'],0)
        self.assertEqual(result['paired_95_percentile'],[0,0])


if __name__=='__main__':
    unittest.main()
