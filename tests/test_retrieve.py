import unittest
import numpy as np
from pyexotransmit.retrieve import Retriever
from pyexotransmit.fit_info import FitInfo

class TestRetriever(unittest.TestCase):
    def setUp(self):
        R_guess = 9.7e7
        T_guess = 1200
        metallicity_guess = 1
        scatt_factor_guess = 1
        cloudtop_P_guess = 1e5
        
        min_wavelength, max_wavelength, self.depths, self.errors = np.loadtxt("tests/testing_data/hd209458b_transit_depths", unpack=True)
        wavelength_bins = np.array([min_wavelength, max_wavelength]).T
        self.wavelength_bins = wavelength_bins

        self.retriever = Retriever()

        self.fit_info = FitInfo({'R': R_guess, 'T': T_guess, 'logZ': np.log10(metallicity_guess), 'CO_ratio': 0.53, 'log_scatt_factor': np.log10(scatt_factor_guess), 'log_cloudtop_P': np.log10(cloudtop_P_guess), 'star_radius': 8.0e8, 'g': 9.311, 'error_multiple': 1})

        self.fit_info.add_fit_param('R', 0.9*R_guess, 1.1*R_guess, 0, np.inf)
        self.fit_info.add_fit_param('T', 0.5*T_guess, 1.5*T_guess, 0, np.inf)
        self.fit_info.add_fit_param('logZ', -1, 3, -1, 3)
        self.fit_info.add_fit_param('CO_ratio', 0.2, 1.5, 0.2, 2.0)
        self.fit_info.add_fit_param('log_cloudtop_P', -1, 4, -np.inf, np.inf)
        #self.fit_info.add_fit_param('log_scatt_factor', 0, 1, 0, 3)
        self.fit_info.add_fit_param('error_multiple', 0.1, 10, 0, np.inf)

    
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
