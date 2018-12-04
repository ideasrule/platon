import unittest
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

import platon
from platon.abundance_getter import AbundanceGetter
from platon.transit_depth_calculator import TransitDepthCalculator
from platon import  __path__
from platon.errors import AtmosphereError
from platon.constants import M_jup, R_sun, R_jup, G, AMU, k_B

class TestTransitDepthCalculator(unittest.TestCase):
    def get_frac_dev(self, logZ, CO_ratio, custom_abundances):
        Rp = 7.14e7
        Mp = 2.0e27
        Rs = 7e8
        T = 1200
        depth_calculator = TransitDepthCalculator()
        wavelengths, transit_depths = depth_calculator.compute_depths(
            Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio,
            custom_abundances=custom_abundances, cloudtop_pressure=1e4)

        # This ExoTransmit run is done without SH, since it's not present in
        # GGchem
        ref_wavelengths, ref_depths = np.loadtxt("tests/testing_data/hot_jupiter_spectra.dat", unpack=True, skiprows=2)
        ref_depths /= 100

        frac_dev = np.abs(ref_depths - transit_depths) / ref_depths

        '''plt.plot(wavelengths, transit_depths, label="platon")
        plt.plot(ref_wavelengths, ref_depths, label="ExoTransmit")
        plt.legend()

        plt.figure()
        plt.plot(np.log10(frac_dev))
        plt.show()'''

        return frac_dev

    def test_custom_file(self):
        custom_abundances = AbundanceGetter.from_file("tests/testing_data/abund_1Xsolar_cond.dat")
        custom_abundances["SH"] *= 0
        frac_dev = self.get_frac_dev(None, None, custom_abundances)

        self.assertLess(np.percentile(frac_dev, 95), 0.03)
        self.assertLess(np.max(frac_dev),  0.07)

    def test_ggchem(self):
        frac_dev = self.get_frac_dev(0, 0.53, None)

        #Some, but very few, individual lines are highly discrepant
        self.assertLess(np.percentile(frac_dev, 95), 0.03)
        self.assertLess(np.max(frac_dev), 0.07)

    def test_unbound_atmosphere(self):
        Rp = 6.378e6
        Mp = 5.97e20 # Note how low this is--10^-4 Earth masses!
        Rs = 6.97e8
        T = 300
        depth_calculator = TransitDepthCalculator()
        with self.assertRaises(AtmosphereError):
            wavelengths, transit_depths = depth_calculator.compute_depths(
                Rs, Mp, Rp, T, logZ=0.2, CO_ratio=1.1, T_star=6100)

    def test_bin_wavelengths(self):
        Rp = 7.14e7
        Mp = 7.49e26
        Rs = 7e8
        T = 1200
        depth_calculator = TransitDepthCalculator()
        bins = np.array([[0.4,0.6], [1,1.1], [1.2,1.4], [3.2,4], [5,6]])
        bins *= 1e-6
        depth_calculator.change_wavelength_bins(bins)

        wavelengths, transit_depths = depth_calculator.compute_depths(
            Rs, Mp, Rp, T, logZ=0.2, CO_ratio=1.1, T_star=6100)
        self.assertEqual(len(wavelengths), len(bins))
        self.assertEqual(len(transit_depths), len(bins))

        wavelengths, transit_depths = depth_calculator.compute_depths(
            Rs, Mp, Rp, T, logZ=0.2, CO_ratio=1.1, T_star=12000)
        self.assertEqual(len(wavelengths), len(bins))
        self.assertEqual(len(transit_depths), len(bins))

    def test_power_law_haze(self):
        Rs = R_sun     
        Mp = M_jup     
        Rp = R_jup      
        T = 1200
        abundances = AbundanceGetter().get(0, 0.53)
        for key in abundances:
            abundances[key] *= 0
        abundances["H2"] += 1
        depth_calculator = TransitDepthCalculator()
        wavelengths, transit_depths, info_dict = depth_calculator.compute_depths(
            Rs, Mp, Rp, T, logZ=None, CO_ratio=None, cloudtop_pressure=np.inf,
            custom_abundances = abundances,
            add_gas_absorption=False, add_collisional_absorption=False, full_output=True)
                
        g = G * Mp / Rp**2
        H = k_B * T / (2 * AMU * g)
        gamma = 0.57721
        polarizability = 0.8059e-30
        sigma = 128. * np.pi**5/3 * polarizability**2 / depth_calculator.lambda_grid**4
        kappa = sigma / (2 * AMU)

        P_surface = 1e8
        R_surface = info_dict["radii"][-1]
        tau_surface = P_surface/g * np.sqrt(2*np.pi*R_surface/H) * kappa
        analytic_R = R_surface + H*(gamma + np.log(tau_surface) + scipy.special.expn(1, tau_surface))
        
        analytic_depths = analytic_R**2 / Rs**2
        
        ratios = analytic_depths / transit_depths
        relative_diffs = np.abs(ratios - 1)
        self.assertTrue(np.all(relative_diffs[wavelengths < 1e-6] < 0.001))

        
    def test_bounds_checking(self):
        Rp = 7.14e7
        Mp = 7.49e26
        Rs = 7e8
        T = 1200
        logZ = 0
        CO_ratio = 1.1
        calculator = TransitDepthCalculator()
        
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, 199, logZ=logZ, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, 3001, logZ=logZ, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=-1.1, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=3.1, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=0.01)
            
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=11)

        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=1e-4)
        
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=1.1e8)

        # Infinity should be fine
        calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=np.inf)
       
            
        
if __name__ == '__main__':
    unittest.main()
