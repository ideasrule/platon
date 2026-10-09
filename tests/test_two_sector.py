import unittest

import numpy as np
from scipy.special import expn

from platon.constants import AU, G, M_jup, R_jup, R_sun
from platon.fit_info import FitInfo
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator


def isothermal(T):
    return Profile.isothermal(T)


# Star and orbit for Guillot profiles: beta = 1 gives T_irr ~ 1290 K
T_STAR = 6000
A_ORBIT = 0.05 * AU


def guillot(beta, log_gamma, log_k_th=-2, T_int=150, Rs=R_sun):
    return Profile.guillot(T_STAR, Rs, A_ORBIT, M_jup, R_jup, beta,
                           log_k_th, log_gamma, T_int)


class TestTwoSectorTypes(unittest.TestCase):
    def test_guillot_equation_and_units(self):
        beta = 1.1
        log_gamma = -1
        log_k_th = -3  # m^2/kg
        T_int = 150
        profile = Profile.guillot(
            T_STAR, R_sun, A_ORBIT, M_jup, R_jup, beta, log_k_th, log_gamma,
            T_int)

        T_irr = beta * T_STAR * np.sqrt(R_sun / (2 * A_ORBIT))
        gamma = 10**log_gamma
        tau = profile.pressures * 10**log_k_th / (G * M_jup / R_jup**2)
        incoming = 2 / 3 + 2 / (3 * gamma) * (
            1 + (gamma * tau / 2 - 1) * np.exp(-gamma * tau))
        incoming += 2 * gamma / 3 * (1 - tau**2 / 2) * \
            expn(2, gamma * tau)
        expected = (
            3 / 4 * T_int**4 * (tau + 2 / 3) +
            3 / 4 * T_irr**4 * incoming)**0.25

        np.testing.assert_allclose(profile.temperatures, expected)
        self.assertEqual(profile.profile_type, "guillot")
        self.assertTrue(np.all(np.isfinite(profile.temperatures)))

    def test_guillot_is_single_channel_radiative_solution(self):
        profile = guillot(1.1, -1, log_k_th=-3)
        radiative = Profile.radiative_solution(
            T_STAR, R_sun, A_ORBIT, M_jup, R_jup, 1.1, -3, -1, None, 0, 150)
        np.testing.assert_allclose(
            profile.temperatures, radiative.temperatures, rtol=1e-12)

    def test_guillot_retrieval_defaults_and_reconstruction(self):
        model = TwoSectorTerminator(
            TerminatorSector(guillot(0.9, -1.2), 1e3),
            TerminatorSector(guillot(1.25, -0.6), 1e6), 0.4)
        defaults = model.retrieval_defaults()
        self.assertEqual(defaults["sector1.beta"], 0.9)
        self.assertEqual(defaults["sector2.beta"], 1.25)
        self.assertEqual(defaults["a"], A_ORBIT)

        rebuilt = model.from_params(defaults)
        for original, sector in ((model.cold, rebuilt.cold),
                                 (model.hot, rebuilt.hot)):
            np.testing.assert_array_equal(
                sector.profile.temperatures, original.profile.temperatures)

        # Cold and hot are ordered by temperature, and must share the star and orbit
        with self.assertRaises(ValueError):
            TwoSectorTerminator(TerminatorSector(guillot(1.25, -1)),
                                TerminatorSector(guillot(0.9, -1)))
        with self.assertRaises(ValueError):
            TwoSectorTerminator(TerminatorSector(guillot(0.9, -1)),
                                TerminatorSector(guillot(1.25, -1, Rs=0.9 * R_sun)))

    def test_retrieval_defaults_and_reconstruction(self):
        model = TwoSectorTerminator(
            TerminatorSector(isothermal(900), 1e3, 10, 6),
            TerminatorSector(isothermal(1400), 1e6, 0.1, 2),
            0.4)
        defaults = model.retrieval_defaults()
        rebuilt = model.from_params(defaults)

        self.assertEqual(defaults["sector1.T"], 900)
        self.assertEqual(defaults["sector2.T"], 1400)
        self.assertEqual(rebuilt.cold_fraction, 0.4)
        self.assertEqual(rebuilt.cold.cloudtop_pressure, 1e3)
        self.assertEqual(rebuilt.hot.scattering_factor, 0.1)

    def test_cold_and_hot_are_ordered(self):
        with self.assertRaises(ValueError):
            TwoSectorTerminator(
                TerminatorSector(isothermal(1500)),
                TerminatorSector(isothermal(1000)))

    def test_ordered_uniform_prior(self):
        model = TwoSectorTerminator(
            TerminatorSector(isothermal(900)),
            TerminatorSector(isothermal(1400)))
        fit_info = FitInfo(model.retrieval_defaults())
        fit_info.add_ordered_uniform_fit_params(
            "sector1.T", "sector2.T", 500, 2000)

        transformed = fit_info._from_unit_interval_array([0.9, 0.1])
        self.assertLess(transformed[0], transformed[1])
        walkers = fit_info._generate_rand_param_arrays(100)
        self.assertTrue(np.all(walkers[:, 0] <= walkers[:, 1]))
        self.assertFalse(fit_info._within_limits([1500, 1000]))
        self.assertEqual(fit_info._ln_prior([1500, 1000]), -np.inf)

    def test_retrieval_configuration(self):
        try:
            from platon.combined_retriever import CombinedRetriever
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        model = TwoSectorTerminator(
            TerminatorSector(isothermal(900), 1e3, 10, 6),
            TerminatorSector(isothermal(1400), 1e6, 1, 4))
        fit_info = CombinedRetriever.get_default_fit_info(
            R_sun, M_jup, R_jup, T=None, transit_terminator=model)
        fit_info.add_uniform_fit_param("sector1.T", 500, 2000)
        fit_info.add_uniform_fit_param("sector2.T", 500, 2000)
        fit_info.add_uniform_fit_param("sector1.fraction", 0, 1)

        self.assertIs(
            fit_info.all_params["transit_terminator"].best_guess, model)
        self.assertIn("sector1.log_cloudtop_P", fit_info.all_params)
        self.assertIn("sector2.log_scatt_factor", fit_info.all_params)

    def test_colder_sector_is_labelled_cold(self):
        from platon.combined_retriever import CombinedRetriever
        from platon.terminator import label_by_temperature
        model = TwoSectorTerminator(
            TerminatorSector(isothermal(900), 1e3, 10, 6),
            TerminatorSector(isothermal(1400), 1e6, 0.1, 2), 0.3)
        params = model.retrieval_defaults()
        # sector1 hotter: the whole sector moves, with its share
        params.update({"sector1.T": 1600., "sector2.T": 800.})
        rebuilt = model.from_params(params)
        self.assertEqual(rebuilt.cold.profile.profile_params["T"], 800.)
        self.assertEqual(rebuilt.cold.cloudtop_pressure, 1e6)
        self.assertEqual(rebuilt.hot.scattering_factor, 10)
        self.assertAlmostEqual(rebuilt.cold_fraction, 0.7)

        fit_info = CombinedRetriever.get_default_fit_info(
            R_sun, M_jup, R_jup, T=None, transit_terminator=model)
        for name in ("sector1.T", "sector2.T"):
            fit_info.add_uniform_fit_param(name, 300, 3000)
        fit_info.add_uniform_fit_param("sector1.fraction", 0, 1)
        fit_info.add_uniform_fit_param("sector1.log_cloudtop_P", 0, 7)
        fit_info.add_uniform_fit_param("sector2.log_cloudtop_P", 0, 7)
        names, labelled = label_by_temperature(
            fit_info, [[800., 2000., .2, 2., 5.], [2500., 1000., .9, 4., 6.]])
        self.assertEqual(names, ["cold.T", "hot.T", "cold_fraction",
                                 "cold.log_cloudtop_P", "hot.log_cloudtop_P"])
        np.testing.assert_allclose(labelled, [[800., 2000., .2, 2., 5.],
                                              [1000., 2500., .1, 6., 4.]])

    def test_guillot_sectors_are_compared_by_temperature_not_beta(self):
        from platon.terminator import sector_temperature
        # Higher beta, but a larger log_gamma deposits the starlight higher up
        low_beta, high_beta = guillot(1.0, -2.5), guillot(1.05, 0.5)
        self.assertGreater(sector_temperature(low_beta), sector_temperature(high_beta))
        model = TwoSectorTerminator(TerminatorSector(high_beta), TerminatorSector(low_beta))
        params = model.retrieval_defaults()
        params.update({"sector1.beta": 1.0, "sector1.log_gamma": -2.5,
                       "sector2.beta": 1.05, "sector2.log_gamma": 0.5})
        self.assertEqual(model.from_params(params).cold.profile.profile_params["beta"], 1.05)


class TestTwoSectorForwardModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Rs = R_sun
        cls.Mp = M_jup
        cls.Rp = R_jup
        cls.calculator = TransitDepthCalculator()
        cls.calculator.change_wavelength_bins(
            1e-6 * np.array([[0.8, 1.0], [1.1, 1.4], [3.5, 4.5]]))

    def test_weighted_depths_and_full_output(self):
        cold = TerminatorSector(isothermal(900), 1e3, 100, 6)
        hot = TerminatorSector(isothermal(1400), 1e6, 0.1, 2)
        model = TwoSectorTerminator(cold, hot, 0.3)

        _, cold_depths, cold_info = self.calculator.compute_depths(
            cold.profile, self.Rs, self.Mp, self.Rp,
            cloudtop_pressure=cold.cloudtop_pressure,
            scattering_factor=cold.scattering_factor,
            scattering_slope=cold.scattering_slope, full_output=True)
        _, hot_depths, hot_info = self.calculator.compute_depths(
            hot.profile, self.Rs, self.Mp, self.Rp,
            cloudtop_pressure=hot.cloudtop_pressure,
            scattering_factor=hot.scattering_factor,
            scattering_slope=hot.scattering_slope, full_output=True)
        _, depths, info = self.calculator.compute_depths(
            model, self.Rs, self.Mp, self.Rp, full_output=True)

        np.testing.assert_allclose(
            depths, 0.3 * cold_depths + 0.7 * hot_depths)
        np.testing.assert_allclose(
            info["unbinned_depths"],
            0.3 * cold_info["unbinned_depths"] +
            0.7 * hot_info["unbinned_depths"])
        self.assertEqual(info["cold_fraction"], 0.3)
        self.assertEqual(set(info["sectors"]), {"cold", "hot"})
        self.assertFalse(np.array_equal(
            info["sectors"]["cold"]["radii"],
            info["sectors"]["hot"]["radii"]))

    def test_identical_and_endpoint_sectors(self):
        sector = TerminatorSector(isothermal(1100), 1e4, 3, 4)
        _, expected, _ = self.calculator.compute_depths(
            sector.profile, self.Rs, self.Mp, self.Rp,
            cloudtop_pressure=sector.cloudtop_pressure,
            scattering_factor=sector.scattering_factor)
        for fraction in (0, 0.4, 1):
            model = TwoSectorTerminator(sector, sector, fraction)
            _, actual, _ = self.calculator.compute_depths(
                model, self.Rs, self.Mp, self.Rp)
            np.testing.assert_allclose(actual, expected)

    def test_guillot_sectors_and_cloud_fraction_error(self):
        cold_profile = guillot(0.9, -1)
        hot_profile = guillot(1.25, -0.5)
        model = TwoSectorTerminator(
            TerminatorSector(cold_profile),
            TerminatorSector(hot_profile), 0.5)

        _, depths, _ = self.calculator.compute_depths(
            model, self.Rs, self.Mp, self.Rp)
        self.assertTrue(np.all(np.isfinite(depths)))
        with self.assertRaises(ValueError):
            self.calculator.compute_depths(
                model, self.Rs, self.Mp, self.Rp, cloud_fraction=0.5)

    def test_guillot_sectors_share_quench_pressure(self):
        cold_profile = guillot(0.85, -1.2)
        hot_profile = guillot(1.3, -0.4)
        cold = TerminatorSector(cold_profile)
        hot = TerminatorSector(hot_profile)
        model = TwoSectorTerminator(cold, hot, 0.4)
        P_quench = 1e5

        _, cold_depths, _ = self.calculator.compute_depths(
            cold_profile, self.Rs, self.Mp, self.Rp, P_quench=P_quench)
        _, hot_depths, _ = self.calculator.compute_depths(
            hot_profile, self.Rs, self.Mp, self.Rp, P_quench=P_quench)
        _, depths, info = self.calculator.compute_depths(
            model, self.Rs, self.Mp, self.Rp,
            P_quench=P_quench, full_output=True)

        # The forward model runs in float32 on the GPU, and the two-sector
        # call compiles its own kernels, which can round differently from
        # those of the single-sector calls (observed: ~1 float32 ulp)
        np.testing.assert_allclose(
            depths, 0.4 * cold_depths + 0.6 * hot_depths, rtol=1e-6)
        for sector_info in info["sectors"].values():
            above = sector_info["P_profile"] <= P_quench
            for abundance in sector_info["atm_abundances"].values():
                np.testing.assert_allclose(
                    abundance[above], abundance[above][0], rtol=1e-5)

    def test_retrieval_likelihood_with_fixed_and_free_fraction(self):
        try:
            from platon.combined_retriever import CombinedRetriever
        except ModuleNotFoundError as error:
            self.skipTest(str(error))

        model = TwoSectorTerminator(
            TerminatorSector(isothermal(900), 1e3, 10, 6),
            TerminatorSector(isothermal(1400), 1e6, 1, 4), 0.5)
        _, depths, _ = self.calculator.compute_depths(
            model, self.Rs, self.Mp, self.Rp)
        errors = np.full_like(depths, 1e-5)
        retriever = CombinedRetriever()

        for fit_fraction in (False, True):
            fit_info = retriever.get_default_fit_info(
                self.Rs, self.Mp, self.Rp, T=None,
                transit_terminator=model)
            fit_info.add_uniform_fit_param("sector1.T", 500, 2000)
            fit_info.add_uniform_fit_param("sector2.T", 500, 2000)
            if fit_fraction:
                fit_info.add_uniform_fit_param("sector1.fraction", 0, 1)
            retriever._validate_params(fit_info, self.calculator)
            params = np.array([
                fit_info.all_params[name].best_guess
                for name in fit_info.fit_param_names])
            log_likelihood = retriever._ln_like(
                params, self.calculator, None, fit_info,
                depths, errors, None, None)
            self.assertTrue(np.isfinite(log_likelihood))

    def test_guillot_retrieval_likelihood(self):
        try:
            from platon.combined_retriever import CombinedRetriever
        except ModuleNotFoundError as error:
            self.skipTest(str(error))

        model = TwoSectorTerminator(
            TerminatorSector(guillot(0.9, -1.2), 1e3),
            TerminatorSector(guillot(1.25, -0.6), 1e6), 0.5)
        _, depths, _ = self.calculator.compute_depths(
            model, self.Rs, self.Mp, self.Rp)
        errors = np.full_like(depths, 1e-5)
        retriever = CombinedRetriever()

        # The star and orbit come from the template; T_star and a are not
        # passed here
        fit_info = retriever.get_default_fit_info(
            self.Rs, self.Mp, self.Rp, T=None, transit_terminator=model)
        self.assertEqual(fit_info._get("T_star"), T_STAR)
        self.assertEqual(fit_info._get("a"), A_ORBIT)
        fit_info.add_uniform_fit_param("sector1.beta", 0.5, 1.5)
        fit_info.add_uniform_fit_param("sector2.beta", 0.5, 1.5)
        fit_info.add_uniform_fit_param("sector1.log_gamma", -3, 1)
        retriever._validate_params(fit_info, self.calculator)
        params = np.array([
            fit_info.all_params[name].best_guess
            for name in fit_info.fit_param_names])
        log_likelihood = retriever._ln_like(
            params, self.calculator, None, fit_info,
            depths, errors, None, None)
        self.assertTrue(np.isfinite(log_likelihood))

        with self.assertRaises(ValueError):
            retriever.get_default_fit_info(
                0.9 * self.Rs, self.Mp, self.Rp, T=None,
                transit_terminator=model)


if __name__ == "__main__":
    unittest.main()
