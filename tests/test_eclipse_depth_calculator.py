import unittest
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter

import platon
from platon.abundance_getter import AbundanceGetter
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.constants import R_sun, M_jup, R_jup, h, c, k_B, AU
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

        '''plt.loglog(1e6 * wavelengths, 1e-3 * blackbody, label="Blackbody")
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
        plt.show()'''
        
        # Should be exact, but in practice isn't, due to our discretization
        self.assertLess(np.percentile(np.abs(rel_diffs), 50), 0.02)
        self.assertLess(np.percentile(np.abs(rel_diffs), 99), 0.05)
        self.assertLess(np.max(np.abs(rel_diffs)), 0.1)

        blackbody_star = np.pi * 2*h*c**2/wavelengths**5/(np.exp(h*c/wavelengths/k_B/Ts) - 1)        
        approximate_depths = blackbody / blackbody_star * (R_jup/R_sun)**2
        # Not expected to be very accurate because the star is not a blackbody
        self.assertLess(np.median(np.abs(approximate_depths - depths)/approximate_depths), 0.2)

    def test_ktables_unbinned(self):
        profile = Profile()
        profile.set_from_radiative_solution(
            5052, 0.75 * R_sun, 0.03142 * AU, 1.129 * M_jup, 1.115 * R_jup,
            0.983, -1.77, -0.44, -0.56, 0.23)
        xsec_calc = EclipseDepthCalculator(method="xsec")
        ktab_calc = EclipseDepthCalculator(method="ktables")
        xsec_wavelengths, xsec_depths = xsec_calc.compute_depths(
            profile, 0.75 * R_sun, 1.129 * M_jup, 1.115 * R_jup,
            5052)
        N = 10
        smoothed_xsec_wavelengths = uniform_filter(xsec_wavelengths, N)[::N]
        smoothed_xsec_depths = uniform_filter(xsec_depths, N)[::N]
        
        ktab_wavelengths, ktab_depths = ktab_calc.compute_depths(
            profile, 0.75 * R_sun, 1.129 * M_jup, 1.115 * R_jup,
            5052)
        rel_diffs = np.abs(ktab_depths - smoothed_xsec_depths[:-1])/ ktab_depths
        self.assertTrue(np.median(rel_diffs) < 0.05)
        
    def test_ktables_binned(self):
        wavelengths = np.exp(np.arange(np.log(0.31e-6), np.log(29e-6), 1./20))
        wavelengths = np.append(wavelengths[0:20], wavelengths[50:90])
        wavelength_bins = np.array([wavelengths[0:-1], wavelengths[1:]]).T

        profile = Profile()
        profile.set_from_radiative_solution(
            5052, 0.75 * R_sun, 0.03142 * AU, 1.129 * M_jup, 1.115 * R_jup,
            0.983, -1.77, -0.44, -0.56, 0.23)
        xsec_calc = EclipseDepthCalculator(method="xsec")
        xsec_calc.change_wavelength_bins(wavelength_bins)
        ktab_calc = EclipseDepthCalculator(method="ktables")
        ktab_calc.change_wavelength_bins(wavelength_bins)
        xsec_wavelengths, xsec_depths = xsec_calc.compute_depths(
            profile, 0.75 * R_sun, 1.129 * M_jup, 1.115 * R_jup,
            5052)
        ktab_wavelengths, ktab_depths = ktab_calc.compute_depths(
            profile, 0.75 * R_sun, 1.129 * M_jup, 1.115 * R_jup,
            5052)
        rel_diffs = np.abs(ktab_depths - xsec_depths)/ ktab_depths
        self.assertTrue(np.median(rel_diffs) < 0.03)
        self.assertTrue(np.percentile(rel_diffs, 95) < 0.15)
        self.assertTrue(np.max(rel_diffs) < 0.3)
        
        '''print(np.median(rel_diffs), np.percentile(rel_diffs, 95), np.max(rel_diffs))
        plt.loglog(xsec_wavelengths, xsec_depths)
        plt.loglog(ktab_wavelengths, ktab_depths)
        plt.figure()
        plt.semilogx(ktab_wavelengths, rel_diffs)
        plt.show()'''

if __name__ == '__main__':
    unittest.main()
