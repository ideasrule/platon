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



if __name__ == '__main__':
    unittest.main()
