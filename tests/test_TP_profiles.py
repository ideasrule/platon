import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from platon.TP_profile import Profile
from platon.constants import M_jup, R_jup, R_sun, AU
from platon.params import NUM_LAYERS

class TestTPProfile(unittest.TestCase):
    def test_isothermal(self):
        profile = Profile.isothermal(1300)
        self.assertEqual(len(profile.pressures), NUM_LAYERS)
        self.assertEqual(len(profile.temperatures), NUM_LAYERS)
        self.assertTrue(np.all(profile.temperatures == 1300))

    def test_parametric(self):
        T0 = 1300
        P1 = 1e-3
        alpha1 = 0.3
        alpha2 = 0.5
        P3 = 1e4
        T3 = 2000
        profile = Profile.parametric(T0, P1, alpha1, alpha2, P3, T3)
        P = profile.pressures
        T = profile.temperatures
        P0 = np.min(P)

        upper = P < P1
        self.assertTrue(np.allclose(
            T[upper], T0 + np.log(P[upper] / P0)**2 / alpha1**2))
        self.assertTrue(np.all(T[P >= P3] == T3))

        # The middle layer is quadratic in ln(P) with curvature 1/alpha2**2,
        # and joins the upper layer at P1 and the isothermal layer at P3
        middle = (P >= P1) & (P < P3)
        quadratic = np.polyfit(np.log(P[middle]), T[middle], 2)
        self.assertTrue(np.isclose(quadratic[0], 1 / alpha2**2))
        T1 = T0 + np.log(P1 / P0)**2 / alpha1**2
        self.assertTrue(np.isclose(np.polyval(quadratic, np.log(P1)), T1))
        self.assertTrue(np.isclose(np.polyval(quadratic, np.log(P3)), T3))

    def test_raw_arrays(self):
        pressures = [1, 10, 100]
        temperatures = [500, 600, 700]
        profile = Profile(pressures, temperatures)
        self.assertTrue(np.array_equal(profile.pressures, pressures))
        self.assertTrue(np.array_equal(profile.temperatures, temperatures))
        self.assertIsNone(profile.profile_type)
        with self.assertRaises(ValueError):
            Profile([1, 10], [500, 600, 700])

        
    def test_suffixed_params(self):
        params = {"T": 1000, "T_transit": 1500}
        profile = Profile.from_params_dict(
            "isothermal", params, suffix="_transit")
        self.assertTrue(np.all(profile.temperatures == 1500))

        profile = Profile.from_params_dict("isothermal", params)
        self.assertTrue(np.all(profile.temperatures == 1000))

        # Unsuffixed values are the fallback when no suffixed version exists
        profile = Profile.from_params_dict("isothermal", {"T": 1000},
                                           suffix="_transit")
        self.assertTrue(np.all(profile.temperatures == 1000))

    def test_radiative_solution(self):
        # Parameters from Table 1 of http://iopscience.iop.org/article/10.1088/0004-637X/775/2/137/pdf
        p = Profile.radiative_solution(5040, 0.756*R_sun, 0.031 * AU, 0.885*M_jup, R_jup, 1, np.log10(3e-3), np.log10(1.58e-1), np.log10(1.58e-1), 0.5, 100)

        # Compare to Figure 2 of aforementioned paper
        is_upper_atm = np.logical_and(p.pressures > 0.1, p.pressures < 1e3)
        self.assertTrue(np.all(p.temperatures[is_upper_atm] > 1000))
        self.assertTrue(np.all(p.temperatures[is_upper_atm] < 1100))

        is_lower_atm = np.logical_and(p.pressures > 1e5, p.pressures < 3e6)
        self.assertTrue(np.all(p.temperatures[is_lower_atm] > 1600))
        self.assertTrue(np.all(p.temperatures[is_lower_atm] < 1700))

        self.assertTrue(np.all(np.diff(p.temperatures) > 0))


if __name__ == '__main__':
    unittest.main()
    
