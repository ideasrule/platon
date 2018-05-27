import unittest

import numpy as np
import matplotlib.pyplot as plt

import eos_reader
from transit_depth_calculator import TransitDepthCalculator

class TestTransitDepthCalculator(unittest.TestCase):
    def test_EOS_file(self):    
        depth_calculator = TransitDepthCalculator(7e8, 9.8)
        custom_abundances = eos_reader.get_abundances("EOS/eos_1Xsolar_cond.dat")
        wavelengths, transit_depths = depth_calculator.compute_depths(6.4e6, 800, custom_abundances=custom_abundances, high_P=1.014e5)

        ref_wavelengths, ref_depths = np.loadtxt("testing_data/ref_spectra.dat", unpack=True, skiprows=2)
        ref_depths /= 100

        frac_dev = np.abs(ref_depths - transit_depths) / ref_depths
        self.assertTrue(np.all(frac_dev < 0.01))

    def test_ggchem(self):
        depth_calculator = TransitDepthCalculator(7e8, 9.8, include_condensates=True)
        wavelengths, transit_depths = depth_calculator.compute_depths(6.4e6, 800, 0, 0.53, high_P=1.014e5)

        ref_wavelengths, ref_depths = np.loadtxt("testing_data/ref_spectra.dat", unpack=True, skiprows=2)
        ref_depths /= 100

        frac_dev = np.abs(ref_depths - transit_depths) / ref_depths

        #Some, but very few, individual lines are highly discrepant
        self.assertLess(np.percentile(frac_dev, 95), 0.01)

        #These lines are sodium and potassium lines; abundances are discrepant
        #between GGchem and Eliza's code. We mask out those lines

        frac_dev[np.logical_and(wavelengths > 403e-9, wavelengths < 406e-9)] = 0
        frac_dev[np.logical_and(wavelengths > 580e-9, wavelengths < 600e-9)] = 0
        frac_dev[np.logical_and(wavelengths > 750e-9, wavelengths < 780e-9)] = 0

        #This is a HCl line.  In this case, ExoTransmit does not predict
        #HCl abundances correctly
        frac_dev[np.logical_and(wavelengths > 3.76e-6, wavelengths < 3.78e-6)] = 0
        self.assertLess(np.max(frac_dev), 0.02)
        


if __name__ == '__main__':
    unittest.main()
