import unittest
import numpy as np

from platon._params import _UniformParam, _GaussianParam

class TestParams(unittest.TestCase):
    def test_uniform_param(self):
        p = _UniformParam(2, 1, 10, 1.5, 3)
        self.assertTrue(p.within_limits(1.1))
        self.assertTrue(p.within_limits(9))
        self.assertFalse(p.within_limits(15))

        self.assertEqual(p.ln_prior(1.1), 0)
        self.assertEqual(p.ln_prior(100), -np.inf)

        self.assertEqual(p.from_unit_interval(0.5), 5.5)
        self.assertEqual(p.from_unit_interval(0.2), 2.8)

    def test_gaussian_param(self):
        mu = 10.0
        std = 2.0
        
        p = _GaussianParam(mu, std, None, None)
        self.assertTrue(p.within_limits(-100))
        self.assertTrue(p.within_limits(100))

        x = mu
        prob_density = np.exp(-(x-mu)**2/2/std**2)/np.sqrt(2*np.pi*std**2)
        self.assertEqual(np.log(prob_density), p.ln_prior(x))

        x = 5
        prob_density = np.exp(-(x-mu)**2/2/std**2)/np.sqrt(2*np.pi*std**2)
        self.assertEqual(np.log(prob_density), p.ln_prior(x))

        self.assertEqual(p.from_unit_interval(0.5), 10)

        diff = np.abs(p.from_unit_interval(0.01) - (mu - 2.32635*std))
        self.assertTrue(diff < 1e-3)

        diff = np.abs(p.from_unit_interval(0.025) - (mu - 1.95996*std))
        self.assertTrue(diff < 1e-3)

        diff = np.abs(p.from_unit_interval(0.84135) - (mu + std))
        self.assertTrue(diff < 1e-3)

if __name__ == '__main__':
    unittest.main()
