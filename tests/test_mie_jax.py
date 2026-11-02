import multiprocessing
import unittest

import numpy as np

import platon._atmosphere_solver as AS
from platon import _mie_jax, _mie_multi_x
from platon._mie_cache import MieCache
from platon.transit_depth_calculator import TransitDepthCalculator


_FORK_ATM = None   # inherited by forked children (not pickled)


def _child_xsec(args):
    ri, part_size = args
    _FORK_ATM._mie_eff_xsec_cache = {}
    _FORK_ATM._mie_cache = MieCache()
    return AS._IN_FORKED_CHILD, \
        _FORK_ATM.get_mie_eff_cross_section(ri, part_size)


class TestMieJax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calc = TransitDepthCalculator()
        wavelengths = np.linspace(0.5, 5, 300) * 1e-6
        cls.calc.change_wavelength_bins(
            np.array([wavelengths[:-1], wavelengths[1:]]).T)
        cls.atm = cls.calc.atm

    def _xsec(self, ri, part_size, use_jax, sigma=0.5):
        old = AS._MIE_JAX
        AS._MIE_JAX = use_jax
        try:
            self.atm._mie_eff_xsec_cache = {}
            self.atm._mie_cache = MieCache()
            return self.atm.get_mie_eff_cross_section(ri, part_size, sigma=sigma)
        finally:
            AS._MIE_JAX = old

    def test_qext_matches_reference(self):
        for x_max in [1.3, 130, 1300]:
            xs = np.exp(np.linspace(np.log(x_max / 1000), np.log(x_max), 300))
            for m in [1.6 - 1e-6j, 1.6 - 0.1j, 1.33]:
                expected = _mie_multi_x.get_Qext(m, xs)
                np.testing.assert_allclose(_mie_jax.get_Qext(m, xs), expected,
                                           rtol=1e-8)

    def test_eff_cross_section_matches_reference(self):
        for part_size in [1e-8, 1e-7, 1e-6, 1e-5]:
            for ri in [1.6 - 1e-6j, 1.6 - 1e-2j, 1.6 - 1j, 1.33]:
                expected = self._xsec(ri, part_size, False)
                result = self._xsec(ri, part_size, True)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_allclose(result, expected, rtol=2e-6)
        for sigma in [0.1, 1.0]:
            np.testing.assert_allclose(
                self._xsec(1.6 - 1e-3j, 1e-6, True, sigma),
                self._xsec(1.6 - 1e-3j, 1e-6, False, sigma), rtol=2e-6)

    def test_fork_pool(self):
        # The parent uses JAX; forked children must fall back to numpy
        global _FORK_ATM
        parent = self._xsec(1.6 - 0.01j, 1e-6, True)
        _FORK_ATM = self.atm
        with multiprocessing.get_context("fork").Pool(2) as pool:
            results = pool.map_async(
                _child_xsec, [(1.6 - 0.01j, 1e-6)] * 2).get(timeout=300)
        for in_child, result in results:
            self.assertTrue(in_child)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, parent, rtol=2e-6)

    def test_cache_bounded_cost(self):
        cache = MieCache()
        xs = np.exp(np.linspace(-5, 5, 100))
        for i in range(30):
            cache.get_and_update(1.6 - 1j * 10.0**(-i / 10), xs,
                                 qext_func=lambda m, x: np.ones(len(x)))
        self.assertEqual(len(cache), 3000)
        # Same m, xs just below cached ones: served from the cache
        calls = []
        result = cache.get_and_update(
            1.6 - 1j * 10.0**(-2.9), xs[1:] * 0.99,
            qext_func=lambda m, x: calls.append(len(x)) or np.zeros(len(x)))
        self.assertEqual(calls, [])
        np.testing.assert_allclose(result, 1)
        # A new m is computed
        cache.get_and_update(
            1.7, xs, qext_func=lambda m, x: calls.append(len(x)) or np.zeros(len(x)))
        self.assertEqual(calls, [100])


if __name__ == '__main__':
    unittest.main()
