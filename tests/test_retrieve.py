import unittest
import numpy as np
import copy
import emcee
import nestle
import matplotlib
matplotlib.use('Agg')

from platon.retriever import Retriever
from platon.fit_info import FitInfo
from platon.constants import R_sun, R_jup, M_jup

class TestRetriever(unittest.TestCase):
    def initialize(self, use_guesses=False):
        min_wavelength, max_wavelength, self.depths, self.errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        wavelength_bins = np.array([min_wavelength, max_wavelength]).T
        self.wavelength_bins = wavelength_bins

        self.retriever = Retriever()

        self.fit_info = Retriever.get_default_fit_info(
            Rs = 1.19 * R_sun, Mp = 0.73 * M_jup, Rp = 1.4 * R_jup, T = 1200,
            logZ = 1, CO_ratio = 0.53,
            log_cloudtop_P = 3,
            log_scatt_factor = 0,
            scatt_slope = 4, error_multiple = 1)

        self.fit_info.add_gaussian_fit_param('Rs', 0.02*R_sun)
        self.fit_info.add_gaussian_fit_param('Mp', 0.04*M_jup)

        if use_guesses:
            self.fit_info.add_uniform_fit_param('R', 0, np.inf, 9e7, 12e7)
            self.fit_info.add_uniform_fit_param('T', 300, 3000, 800, 1800)
            self.fit_info.add_uniform_fit_param('logZ', -1, 3, -1, 3)
            self.fit_info.add_uniform_fit_param('CO_ratio', 0.2, 2.0, 0.2, 1.5)
            self.fit_info.add_uniform_fit_param('log_cloudtop_P', -0.9, 4, 0, 3)
            self.fit_info.add_uniform_fit_param('log_scatt_factor', 0, 3, 0, 1)
            self.fit_info.add_uniform_fit_param('scatt_slope', 0, 10, 1, 5)
            self.fit_info.add_uniform_fit_param('error_multiple', 0, np.inf, 0.1, 10)
        else:
            self.fit_info.add_uniform_fit_param('R', 9e7, 12e7)
            self.fit_info.add_uniform_fit_param('T', 800, 1800)
            self.fit_info.add_uniform_fit_param('logZ', -1, 3)
            self.fit_info.add_uniform_fit_param('CO_ratio', 0.2, 1.5)
            self.fit_info.add_uniform_fit_param('log_cloudtop_P', -0.99, 4)
            self.fit_info.add_uniform_fit_param('log_scatt_factor', 0, 1)
            self.fit_info.add_uniform_fit_param('scatt_slope', 1, 5)
            self.fit_info.add_uniform_fit_param('error_multiple', 0.1, 10)


    def test_emcee(self):
        self.initialize(True)
        nwalkers = 20
        nsteps = 20
        
        retriever = Retriever()
        result = retriever.run_emcee(self.wavelength_bins, self.depths, self.errors, self.fit_info, nsteps=nsteps, nwalkers=nwalkers, include_condensation=False)
        self.assertTrue(isinstance(result, emcee.ensemble.EnsembleSampler))
        self.assertTrue(result.chain.shape, (nwalkers, nsteps, len(self.fit_info.fit_param_names)))
        self.assertTrue(result.lnprobability.shape, (nwalkers, nsteps))
        
        retriever = Retriever()
        result = retriever.run_emcee(self.wavelength_bins, self.depths, self.errors, self.fit_info, nsteps=nsteps, nwalkers=nwalkers, include_condensation=True, plot_best=True)
        self.assertTrue(isinstance(result, emcee.ensemble.EnsembleSampler))
        self.assertEqual(result.chain.shape, (nwalkers, nsteps, len(self.fit_info.fit_param_names)))
        self.assertEqual(result.lnprobability.shape, (nwalkers, nsteps))
                

    def test_multinest(self):
        self.initialize(False)
        retriever = Retriever()
        result = retriever.run_multinest(self.wavelength_bins, self.depths, self.errors, self.fit_info, maxcall=200, include_condensation=False, plot_best=True)
        
        self.assertTrue(isinstance(result, nestle.Result))
        self.assertEqual(result.samples.shape[1], len(self.fit_info.fit_param_names))

        retriever = Retriever()
        retriever.run_multinest(self.wavelength_bins, self.depths, self.errors, self.fit_info, maxiter=20, include_condensation=True)
        self.assertTrue(isinstance(result, nestle.Result))
        self.assertEqual(result.samples.shape[1], len(self.fit_info.fit_param_names))


    def test_bounds_check(self):
        self.initialize(False)
        retriever = Retriever()

        def run_both(name, low_lim, best_guess, high_lim):
            fit_info = copy.deepcopy(self.fit_info)
            fit_info.all_params[name].low_lim = low_lim
            fit_info.all_params[name].best_guess = best_guess
            fit_info.all_params[name].high_lim = high_lim
            with self.assertRaises(ValueError):
                retriever.run_multinest(self.wavelength_bins, self.depths,
                                        self.errors, fit_info, maxiter=10)
                retriever.run_emcee(self.wavelength_bins, self.depths,
                                    self.errors, fit_info, nsteps=10)

        # All of these are invalid inputs
        run_both("T", 299, 1000, 2999)
        run_both("T", 3000, 1000, 300)
        
        run_both("T", 301, 1000, 3001)        
        run_both("T", 301, 1000, 999)

        run_both("logZ", -1.1, 0, 3)
        run_both("logZ", -1, 2, 3.1)        

        run_both("CO_ratio", 0.19, 0.53, 10)
        run_both("CO_ratio", 0.2, 0.53, 10.1)

        run_both("log_cloudtop_P", -4.1, 0, 5)
        run_both("log_cloudtop_P", -4, 2, 5.1) 
        

if __name__ == '__main__':
    unittest.main()
