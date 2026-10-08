import unittest

import numpy as np

from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.jax_eclipse_depth_calculator import (
    EclipseDepthCalculator as JaxEclipseDepthCalculator)
from platon.jax_combined_retriever import CombinedRetriever
from platon.TP_profile import Profile
from platon.constants import R_sun, R_jup, M_jup, AU

# The JAX forward model runs in float32, so agreement with the float64 NumPy
# model is limited to ~1e-6 relative.  Eclipse depths are O(1e-3), i.e. these
# tolerances correspond to well under 0.01 ppm.
REL_TOL = 5e-5

Rs = 0.86 * R_sun
Mp = 0.58 * M_jup
Rp = 1.05 * R_jup
T_star = 5350.0
OPACITIES = ["H2O", "CH4", "CO2", "CO"]


def _bins():
    edges = np.linspace(3.0e-6, 5.0e-6, 41)
    return np.array([[edges[i], edges[i + 1]] for i in range(len(edges) - 1)])


class TestJaxEclipseDepthCalculator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bins = _bins()
        cls.np_calc = EclipseDepthCalculator(
            method="xsec", include_opacities=OPACITIES)
        cls.np_calc.change_wavelength_bins(cls.bins)
        cls.jax_calc = JaxEclipseDepthCalculator(
            method="xsec", include_opacities=OPACITIES)
        cls.jax_calc.change_wavelength_bins(cls.bins)

    def _compare(self, profile, **kwargs):
        wl_np, d_np, _ = self.np_calc.compute_depths(
            profile, Rs, Mp, Rp, T_star, **kwargs)
        wl_jax, d_jax, info = self.jax_calc.compute_depths(
            profile, Rs, Mp, Rp, T_star, full_output=True, **kwargs)

        np.testing.assert_allclose(wl_jax, wl_np, rtol=1e-12)
        np.testing.assert_allclose(d_jax, d_np, rtol=REL_TOL)
        self.assertEqual(len(info["unbinned_eclipse_depths"]),
                         len(info["unbinned_wavelengths"]))
        self.assertTrue(np.all(np.isfinite(d_jax)))
        return d_np, d_jax

    def test_isothermal_clear(self):
        p = Profile()
        p.set_isothermal(1200.0)
        self._compare(p, logZ=0.0, CO_ratio=0.53)

    def test_isothermal_cloudy(self):
        p = Profile()
        p.set_isothermal(1200.0)
        self._compare(p, logZ=0.0, CO_ratio=0.53, cloudtop_pressure=1e3)

    def test_enhanced_scattering(self):
        p = Profile()
        p.set_isothermal(900.0)
        self._compare(p, logZ=0.0, CO_ratio=0.53,
                      scattering_factor=100.0, scattering_slope=6.0)

    def test_parametric_profile(self):
        p = Profile()
        p.set_parametric(900.0, 10**2.4, 2.0, 2.0, 10**6.0, 1600.0)
        self._compare(p, logZ=0.0, CO_ratio=0.53)

    def test_radiative_solution_profile(self):
        p = Profile()
        p.set_from_radiative_solution(
            T_star, Rs, 0.088 * AU, Mp, Rp, 1.0, -2.0, -0.5, -0.8, 0.5)
        self._compare(p, logZ=0.0, CO_ratio=0.53)

    def test_free_vmr(self):
        p = Profile()
        p.set_isothermal(1100.0)
        gases = ["H2O", "CH4", "CO2", "CO", "H2-He"]
        vmrs = [1e-3, 1e-5, 1e-4, 3e-4]
        vmrs.append(1 - sum(vmrs))
        self._compare(p, logZ=None, CO_ratio=None, gases=gases, vmrs=vmrs)

    def test_cloud_changes_depths(self):
        """The cloud-deck branch must actually do something."""
        p = Profile()
        p.set_parametric(900.0, 10**2.4, 2.0, 2.0, 10**6.0, 1600.0)
        _, clear = self._compare(p, logZ=0.0, CO_ratio=0.53)
        _, cloudy = self._compare(p, logZ=0.0, CO_ratio=0.53,
                                  cloudtop_pressure=1e3)
        self.assertGreater(np.max(np.abs(cloudy - clear)) / np.max(clear), 1e-3)


class TestJaxEclipseLikelihood(unittest.TestCase):
    """The retriever's JIT-compiled likelihood must match _ln_like."""

    def test_per_point_lnlike_matches_numpy(self):
        import jax.numpy as jnp

        bins = _bins()
        rng = np.random.default_rng(0)
        depths = 8e-4 + 1e-4 * rng.standard_normal(len(bins))
        errors = np.full(len(bins), 5e-5)

        retriever = CombinedRetriever()
        retriever.params_to_lnlike = {}
        fit_info = retriever.get_default_fit_info(
            Rs=Rs, Mp=Mp, Rp=Rp, logZ=None, CO_ratio=None, T_star=T_star,
            error_multiple=1.0, fit_vmr=True, profile_type="parametric",
            T0=900.0, log_P1=2.4, alpha1=2.0, alpha2=2.0, log_P3=6.0,
            T3=1600.0)
        fit_info.add_uniform_fit_param("T0", 500, 1500)
        fit_info.add_uniform_fit_param("T3", 1000, 2500)
        fit_info.add_gases_vmr(["H2O", "CH4", "CO2", "H2-He"], 1e-10, 0.1)

        eclipse_calc = EclipseDepthCalculator(
            method="xsec", include_opacities=OPACITIES)
        eclipse_calc.change_wavelength_bins(bins)

        setup = retriever._setup_jax_likelihood(
            fit_info, "unittest", None, None, None, None,
            eclipse_calc, bins, depths, errors)

        n_params = len(fit_info.fit_param_names)
        rng2 = np.random.default_rng(3)
        compared = 0
        for _ in range(4):
            cube = np.array([fit_info._from_unit_interval(i, u)
                             for i, u in enumerate(rng2.uniform(size=n_params))])
            per_point_jax = np.array(setup.per_point_lnlike(
                jnp.array(cube, dtype=jnp.float32)))
            per_point_np = retriever._ln_like(
                cube, None, eclipse_calc, fit_info, None, None,
                depths, errors, lnlike_per_point=True)
            if np.isscalar(per_point_np):
                continue
            compared += 1
            self.assertEqual(per_point_jax.shape, per_point_np.shape)
            # Absolute agreement on the summed log-likelihood, scaled by its
            # own magnitude: float32 noise on the depths propagates through
            # (residual / error)^2.
            self.assertLess(
                abs(per_point_jax.sum() - per_point_np.sum()),
                1e-4 * max(abs(per_point_np.sum()), 1.0))
        self.assertGreater(compared, 0)


if __name__ == "__main__":
    unittest.main()
