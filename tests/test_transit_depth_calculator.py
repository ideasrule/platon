import unittest

import numpy as np
import matplotlib.pyplot as plt

from platon.abundance_getter import AbundanceGetter
from platon.transit_depth_calculator import TransitDepthCalculator

class TestTransitDepthCalculator(unittest.TestCase):
    def get_frac_dev(self, logZ, CO_ratio, custom_abundances):
        Rp = 7.14e7
        Mp = 7.49e26
        Rs = 7e8
        T = 1200
        depth_calculator = TransitDepthCalculator(max_P_profile=1.014e5)
        wavelengths, transit_depths = depth_calculator.compute_depths(
            Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio,
            custom_abundances=custom_abundances)

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
        self.assertTrue(np.all(frac_dev < 0.02))

    def test_ggchem(self):
        frac_dev = self.get_frac_dev(0, 0.53, None)

        #Some, but very few, individual lines are highly discrepant
        self.assertLess(np.percentile(frac_dev, 95), 0.02)

        self.assertLess(np.max(frac_dev), 0.03)

    def test_unbound_atmosphere(self):
        Rp = 6.378e6
        Mp = 5.97e20 # Note how low this is--10^-4 Earth masses!
        Rs = 6.97e8
        T = 300
        depth_calculator = TransitDepthCalculator()
        with self.assertRaises(ValueError):
            wavelengths, transit_depths = depth_calculator.compute_depths(
                Rs, Mp, Rp, T, logZ=0.2, CO_ratio=1.1, T_star=6100)

    def test_bin_wavelengths(self):
        Rp = 7.14e7
        Mp = 7.49e26
        Rs = 7e8
        T = 1200
        depth_calculator = TransitDepthCalculator(max_P_profile=1.014e5)
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

    def test_bounds_checking(self):
        Rp = 7.14e7
        Mp = 7.49e26
        Rs = 7e8
        T = 1200
        logZ = 0
        CO_ratio = 1.1
        calculator = TransitDepthCalculator()
        
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, 299, logZ=logZ, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, 3001, logZ=logZ, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=-1.1, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=3.1, CO_ratio=CO_ratio)
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=0.19)
            
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=11)

        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=0.1)
        
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=1.1e5)

        # Infinity should be fine
        calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=np.inf)

        # If min_P_profile or max_P_profile are different, so should the
        # appropriate bounds for cloudtop_pressure
        calculator = TransitDepthCalculator(min_P_profile=1e-2, max_P_profile=1e6)
        calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=1.1e-2)
        calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=1e6)

        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=0.99e-2)
        
        with self.assertRaises(ValueError):
            calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=1.1e6)

            
        
            
        
if __name__ == '__main__':
    unittest.main()
