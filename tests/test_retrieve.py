import unittest
import numpy as np
from platon.retriever import Retriever
from platon.fit_info import FitInfo

class TestRetriever(unittest.TestCase):
    def setUp(self):
        min_wavelength, max_wavelength, self.depths, self.errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        wavelength_bins = np.array([min_wavelength, max_wavelength]).T
        self.wavelength_bins = wavelength_bins

        self.retriever = Retriever()

        self.fit_info = Retriever.get_default_fit_info(
            Rs = 8.0e8, Mp = 7.49e26, Rp = 9.7e7, T = 1200,
            logZ = 1, CO_ratio = 0.53,
            log_cloudtop_P = 5,
            log_scatt_factor = 0,
            scatt_slope = 4, error_multiple = 1)
                                                       
        self.fit_info.add_fit_param('R', 9e7, 12e7, 0, np.inf)
        self.fit_info.add_fit_param('T', 800, 1800, 0, np.inf)
        self.fit_info.add_fit_param('logZ', -1, 3, -1, 3)
        self.fit_info.add_fit_param('CO_ratio', 0.2, 1.5, 0.2, 2.0)
        self.fit_info.add_fit_param('log_cloudtop_P', -1, 4, -np.inf, np.inf)
        self.fit_info.add_fit_param('log_scatt_factor', 0, 1, 0, 3)
        self.fit_info.add_fit_param('scatt_slope', 1, 5, 0, 10)
        self.fit_info.add_fit_param('error_multiple', 0.1, 10, 0, np.inf)
        self.fit_info.add_fit_param('star_radius', 7e8, 9e8, 0, np.inf)
        self.fit_info.add_fit_param('Mp', 6e26, 9e26, 0, np.inf)


    def test_emcee(self):
        retriever = Retriever()
        retriever.run_emcee(self.wavelength_bins, self.depths, self.errors, self.fit_info, nsteps=30, nwalkers=20, include_condensates=False)

        retriever = Retriever()
        retriever.run_emcee(self.wavelength_bins, self.depths, self.errors, self.fit_info, nsteps=30, nwalkers=20, include_condensates=True)


    def test_multinest(self):
        retriever = Retriever()
        retriever.run_multinest(self.wavelength_bins, self.depths, self.errors, self.fit_info, maxiter=100, include_condensates=False)

        retriever = Retriever()
        retriever.run_multinest(self.wavelength_bins, self.depths, self.errors, self.fit_info, maxiter=100, include_condensates=True)

if __name__ == '__main__':
    unittest.main()
