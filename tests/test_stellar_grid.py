import tempfile
import unittest
from pathlib import Path

import numpy as np

from platon._atmosphere_solver import _load_stellar_grid


class TestStellarGrid(unittest.TestCase):
    def test_interpolation_in_logg_feh_and_wavelength(self):
        temps, loggs, fehs = np.array([3000., 4000.]), np.array([4., 5.]), np.array([-1., 0., .5])
        waves = np.array([1., 2., 4.]) * 1e-6
        # Linear in every axis, so linear interpolation is exact
        T, g, z, w = np.meshgrid(temps, loggs, fehs, waves, indexing="ij")
        spectra = 1e10 * (T + 100 * g + 1000 * z) + 1e15 * w
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.npz"
            np.savez(path, temperatures=temps, loggs=loggs, fehs=fehs,
                     wavelengths=waves, spectra=spectra)
            out_temps, out = _load_stellar_grid(path, 4.3, -.2, np.array([1.5e-6, 3e-6]))
        expected = 1e10 * (temps[:, None] + 430 - 200) + 1e15 * np.array([1.5e-6, 3e-6])
        np.testing.assert_array_equal(out_temps, temps)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


    def test_per_visit_heterogeneities(self):
        from platon.combined_retriever import CombinedRetriever
        fit_info = CombinedRetriever.get_default_fit_info(
            7e8, 1.9e27, 7e7, T=1000, T_star=4000, T_het=3000, het_cov_frac=0.05,
            transit_visits={"v1": (0, 2), "v2": (2, 5)})
        params = fit_info._interpret_param_array([])
        # Nothing per-visit: plain scalars
        self.assertEqual(CombinedRetriever._visit_hets(params, 5),
                         dict(T_het=3000, het_cov_frac=0.05, T_het2=None, het2_cov_frac=None))
        fit_info.add_uniform_fit_param("v2.het_cov_frac", 0, 0.5)
        fit_info.add_uniform_fit_param("v1.T_het2", 4000, 6000)
        hets = CombinedRetriever._visit_hets(fit_info._interpret_param_array([0.3, 5000]), 5)
        np.testing.assert_array_equal(hets["het_cov_frac"], [0.05, 0.05, 0.3, 0.3, 0.3])
        np.testing.assert_array_equal(hets["T_het2"], [5000, 5000, 4000, 4000, 4000])
        self.assertEqual(hets["T_het"], 3000)


if __name__ == '__main__':
    unittest.main()
