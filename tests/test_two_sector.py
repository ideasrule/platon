import unittest

import numpy as np
from scipy.special import expn

from platon.constants import G, M_jup, R_jup, R_sun
from platon.fit_info import FitInfo
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator


def isothermal(T):
    profile = Profile()
    profile.set_isothermal(T)
    return profile


class TestTwoSectorTypes(unittest.TestCase):
    def test_guillot_equation_and_units(self):
        T_irr = 1400
        log_gamma = -1
        log_k_th = -2
        T_int = 150
        profile = Profile()
        profile.set_guillot(
            T_irr, log_gamma, log_k_th, T_int, M_jup, R_jup)

        gamma = 10**log_gamma
        tau = profile.pressures * (0.1 * 10**log_k_th) / \
            (G * M_jup / R_jup**2)
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

    def test_retrieval_defaults_and_reconstruction(self):
        model = TwoSectorTerminator(
            TerminatorSector(isothermal(900), 1e3, 10, 6),
            TerminatorSector(isothermal(1400), 1e6, 0.1, 2),
            0.4)
        defaults = model.retrieval_defaults()
        rebuilt = model.from_params(defaults, M_jup, R_jup)

        self.assertEqual(defaults["cold.T"], 900)
        self.assertEqual(defaults["hot.T"], 1400)
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
            "cold.T", "hot.T", 500, 2000)

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
        fit_info.add_ordered_uniform_fit_params(
            "cold.T", "hot.T", 500, 2000)
        fit_info.add_uniform_fit_param("cold_fraction", 0, 1)

        self.assertIs(
            fit_info.all_params["transit_terminator"].best_guess, model)
        self.assertEqual(
            fit_info.ordered_pairs, [("cold.T", "hot.T")])
        self.assertIn("cold.log_cloudtop_P", fit_info.all_params)
        self.assertIn("hot.log_scatt_factor", fit_info.all_params)


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
        cold_profile = Profile()
        hot_profile = Profile()
        cold_profile.set_guillot(1200, -1, -2, 150, self.Mp, self.Rp)
        hot_profile.set_guillot(1600, -0.5, -2, 150, self.Mp, self.Rp)
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
        cold_profile = Profile()
        hot_profile = Profile()
        cold_profile.set_guillot(1100, -1.2, -2, 150, self.Mp, self.Rp)
        hot_profile.set_guillot(1700, -0.4, -2, 150, self.Mp, self.Rp)
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

        np.testing.assert_allclose(
            depths, 0.4 * cold_depths + 0.6 * hot_depths)
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
            fit_info.add_ordered_uniform_fit_params(
                "cold.T", "hot.T", 500, 2000)
            if fit_fraction:
                fit_info.add_uniform_fit_param("cold_fraction", 0, 1)
            retriever._validate_params(fit_info, self.calculator)
            params = np.array([
                fit_info.all_params[name].best_guess
                for name in fit_info.fit_param_names])
            log_likelihood = retriever._ln_like(
                params, self.calculator, None, fit_info,
                depths, errors, None, None)
            self.assertTrue(np.isfinite(log_likelihood))


if __name__ == "__main__":
    unittest.main()
