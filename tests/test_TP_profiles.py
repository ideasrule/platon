import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from platon.TP_profile import Profile
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.constants import M_jup, R_jup, R_sun, AU
from platon.params import NUM_LAYERS

class TestTPProfile(unittest.TestCase):
    def test_isothermal(self):
        profile = Profile()
        profile.set_isothermal(1300)
        self.assertEqual(len(profile.pressures), NUM_LAYERS)
        self.assertEqual(len(profile.temperatures), NUM_LAYERS)
        self.assertTrue(np.all(profile.temperatures == 1300))

    def test_parametric(self):
        profile = Profile()
        P0 = np.min(profile.pressures)
        T0 = 1300
        P1 = 1e-3
        alpha1 = 0.3
        alpha2 = 0.5
        P3 = 1e4
        T3 = 2000
        P2, T2 = profile.set_parametric(T0, P1, alpha1, alpha2, P3, T3)
        self.assertTrue(abs(P3 - P2 * np.exp(alpha2 * (T3 - T2)**0.5)) < 1e-3*P3)
        T1 = np.log(P1/P0)**2/alpha1**2 + T0
        self.assertTrue(abs(P1 - P0*np.exp(alpha1*(T1-T0)**0.5)) < 1e-3*P1)
        self.assertTrue(abs(P1 - P2*np.exp(-alpha2*(T1-T2)**0.5)) < 1e-3*P1)

        
    def test_set_opacity(self):
        # Can't easily test whether this is "right", but at least make sure it
        # runs
        p = Profile()
        p.set_isothermal(1200)
        
        calc = EclipseDepthCalculator()
        wavelengths, depths, info_dict = calc.compute_depths(
            p, R_sun, M_jup, R_jup, 5700, full_output=True)
        p.set_from_opacity(1700, info_dict)
        self.assertTrue(np.all(p.temperatures > 0))
        self.assertTrue(np.all(~np.isnan(p.temperatures)))

        
    def test_radiative_solution(self):
        p = Profile()

        # Parameters from Table 1 of http://iopscience.iop.org/article/10.1088/0004-637X/775/2/137/pdf
        p.set_from_radiative_solution(5040, 0.756*R_sun, 0.031 * AU, 0.885*M_jup, R_jup, 1, np.log10(3e-3), np.log10(1.58e-1), np.log10(1.58e-1), 0.5, 100)

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
    
