import unittest
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import platon
from platon.abundance_getter import AbundanceGetter
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.constants import R_sun, M_jup, R_jup, h, c, k_B
from platon.TP_profile import Profile

class TestEclipseDepthCalculator(unittest.TestCase):
    def test_isothermal(self):
        Ts = 5700
        Tp = 1500
        p = Profile()
        p.set_isothermal(Tp)
        calc = EclipseDepthCalculator()
        wavelengths, depths, info_dict = calc.compute_depths(p, R_sun, M_jup, R_jup, Ts, full_output=True)
                
        blackbody = np.pi * 2*h*c**2/wavelengths**5/(np.exp(h*c/wavelengths/k_B/Tp) - 1)

        rel_diffs = (info_dict["planet_spectrum"] - blackbody)/blackbody

        plt.loglog(1e6 * wavelengths, 1e-3 * blackbody, label="Blackbody")
        plt.loglog(1e6 * wavelengths, 1e-3 * info_dict["planet_spectrum"], label="PLATON")
        plt.xlabel("Wavelength (micron)", fontsize=12)
        plt.ylabel("Planet flux (erg/s/cm$^2$/micron)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.figure()
        plt.semilogx(1e6 * wavelengths, 100 * rel_diffs)
        plt.xlabel("Wavelength (micron)", fontsize=12)
        plt.ylabel("Relative difference (%)", fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Should be exact, but in practice isn't, due to our discretization
        self.assertLess(np.percentile(np.abs(rel_diffs), 50), 0.02)
        self.assertLess(np.percentile(np.abs(rel_diffs), 99), 0.05)
        self.assertLess(np.max(np.abs(rel_diffs)), 0.1)

        blackbody_star = np.pi * 2*h*c**2/wavelengths**5/(np.exp(h*c/wavelengths/k_B/Ts) - 1)        
        approximate_depths = blackbody / blackbody_star * (R_jup/R_sun)**2
        # Not expected to be very accurate because the star is not a blackbody
        self.assertLess(np.median(np.abs(approximate_depths - depths)/approximate_depths), 0.2)


if __name__ == '__main__':
    unittest.main()
