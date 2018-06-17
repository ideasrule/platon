from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import nestle

from .transit_depth_calculator import TransitDepthCalculator
from .fit_info import FitInfo
from .constants import METRES_TO_UM
from ._params import _UniformParam

class Retriever:
    def _validate_params(self, fit_info, calculator):
        # This assumes that the valid parameter space is rectangular, so that
        # the bounds for each parameter can be treated separately. Unfortunately
        # there is no good way to validate Gaussian parameters, which have
        # infinite range.
        for name in fit_info.fit_param_names:
            this_param = fit_info.all_params[name]
            if not isinstance(this_param, _UniformParam):
                continue

            if this_param.best_guess < this_param.low_lim \
               or this_param.best_guess > this_param.high_lim:
                raise ValueError(
                    "Value {} for {} not between low and high limits".format(
                        this_param.best_guess, name))
            if this_param.low_lim >= this_param.high_lim:
                raise ValueError(
                    "low_lim for {} is higher than high_lim".format(name))
            
            for lim in [this_param.low_lim, this_param.high_lim]:
                this_param.best_guess = lim
                calculator._validate_params(
                    fit_info.get("T"),
                    fit_info.get("logZ"),
                    fit_info.get("CO_ratio"),
                    10**fit_info.get("log_cloudtop_P"))

    
    def _ln_prob(self, params, calculator, fit_info, measured_depths,
                measured_errors, plot=False):
        if not fit_info.within_limits(params):
            return -np.inf

        params_dict = fit_info.interpret_param_array(params)
        R = params_dict["R"]
        T = params_dict["T"]
        logZ = params_dict["logZ"]
        CO_ratio = params_dict["CO_ratio"]
        scatt_factor = 10.0**params_dict["log_scatt_factor"]
        scatt_slope = params_dict["scatt_slope"]
        cloudtop_P = 10.0**params_dict["log_cloudtop_P"]
        error_multiple = params_dict["error_multiple"]
        Rs = params_dict["Rs"]
        Mp = params_dict["Mp"]
        T_star = params_dict["T_star"]
        
        if not calculator.is_in_bounds(logZ, CO_ratio, T, cloudtop_P):
            return -np.inf
        if Rs <= 0 or Mp <= 0:
            return -np.inf

        wavelengths, calculated_depths = calculator.compute_depths(
            Rs, Mp, R, T, logZ, CO_ratio,
            scattering_factor=scatt_factor, scattering_slope = scatt_slope,
            cloudtop_pressure=cloudtop_P, T_star=T_star)
        residuals = calculated_depths - measured_depths
        scaled_errors = error_multiple * measured_errors
        ln_prob = -0.5 * np.sum(residuals**2/scaled_errors**2 + np.log(2*np.pi*scaled_errors**2))

        if plot:
            plt.errorbar(METRES_TO_UM*wavelengths, measured_depths, yerr=measured_errors, fmt='.')
            plt.plot(METRES_TO_UM*wavelengths, calculated_depths)
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Transit depth")
            plt.show()
        return fit_info.ln_prior(params) + ln_prob

    def run_emcee(self, wavelength_bins, depths, errors, fit_info, nwalkers=50,
                  nsteps=10000, include_condensates=True,
                  plot_best=False, max_P_profile=1e5):
        '''Runs affine-invariant MCMC to retrieve atmospheric parameters.

        Parameters
        ----------
        wavelength_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start 
            wavelength and wavelength_bins[i][1] is the end wavelength for 
            bin i.
        depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        errors : array_like, length N
            Errors on the aforementioned transit depths
        fit_info : :class:`.FitInfo` object
            Tells the method what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        nwalkers : int, optional
            Number of walkers to use
        nsteps : int, optional
            Number of steps that the walkers should walk for
        include_condensates : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        plot_best : bool, optional
            If True, plots the best fit model with the data
            
        Returns
        -------
        result : EnsembleSampler object
            This returns emcee's EnsembleSampler object.  The most useful
            attributes in this item are result.chain, which is a (W x S X P)
            array where W is the number of walkers, S is the number of steps,
            and P is the number of parameters; and result.lnprobability, a
            (W x S) array of log probabilities.  For your convenience, this
            object also contains result.flatchain, which is a (WS x P) array
            where WS = W x S is the number of samples; and 
            result.flatlnprobability, an array of length WS      
        '''
        
        initial_positions = fit_info.generate_rand_param_arrays(nwalkers)
        calculator = TransitDepthCalculator(max_P_profile=max_P_profile,
            include_condensates=include_condensates)
        calculator.change_wavelength_bins(wavelength_bins)
        self._validate_params(fit_info, calculator)
        
        sampler = emcee.EnsembleSampler(
            nwalkers, fit_info.get_num_fit_params(), self._ln_prob,
            args=(calculator, fit_info, depths, errors))

        for i, result in enumerate(sampler.sample(initial_positions, iterations=nsteps)):
            if (i+1) % 10 == 0:
                print(str(i+1) + "/" + str(nsteps), sampler.lnprobability[0,i], sampler.chain[0,i])
                
        best_params_arr = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
        best_params_dict = fit_info.interpret_param_array(best_params_arr)
        print("Best params", best_params_dict)

        if plot_best:
            self._ln_prob(best_params_arr, calculator, fit_info, depths, errors, plot=True)
        return sampler

    def run_multinest(self, wavelength_bins, depths, errors, fit_info, maxiter=None, include_condensates=True, plot_best=False, max_P_profile=1e5):
        '''Runs nested sampling to retrieve atmospheric parameters.

        Parameters
        ----------
        wavelength_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start 
            wavelength and wavelength_bins[i][1] is the end wavelength for 
            bin i.
        depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        errors : array_like, length N
            Errors on the aforementioned transit depths
        fit_info : :class:`.FitInfo` object
            Tells us what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        maxiter : bool, optional
            If not None, run at most this many iterations of nestled sampling
        include_condensates : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        plot_best : bool, optional
            If True, plots the best fit model with the data

        Returns
        -------
        result : Result object
            This returns the object returned by nestle.sample  The object is
            dictionary-like and has many useful items.  For example,
            result.samples (or alternatively, result["samples"]) are the
            parameter values of each sample, result.weights contains the
            weights, and result.logl contains the log likelihoods.  result.logz
            is the natural logarithm of the evidence.
        '''
                    
        calculator = TransitDepthCalculator(max_P_profile=max_P_profile,
            include_condensates=include_condensates)
        calculator.change_wavelength_bins(wavelength_bins)
        self._validate_params(fit_info, calculator)
        
        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info.from_unit_interval(i, cube[i])
            return new_cube

        def multinest_ln_prob(cube):
            return self._ln_prob(cube, calculator, fit_info, depths, errors)

        def callback(callback_info):
            print(callback_info["it"], callback_info["logz"],
                  transform_prior(callback_info["active_u"][0]))

        result = nestle.sample(
            multinest_ln_prob, transform_prior, fit_info.get_num_fit_params(),
            callback=callback, method='multi', maxiter=maxiter)

        best_params_arr = result.samples[np.argmax(result.logl)]
        best_params_dict = fit_info.interpret_param_array(best_params_arr)
        print("Best params", best_params_dict)
        
        if plot_best:
            self._ln_prob(best_params_arr, calculator, fit_info, depths, errors, plot=True)
        return result


    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T, logZ=0, CO_ratio=0.53,
                             log_cloudtop_P=3, log_scatt_factor=0,
                             scatt_slope=4, error_multiple=1,
                             T_star=None, add_fit_params=False):

        fit_info = FitInfo({'Mp': Mp, 'R': Rp, 'T': T, 'logZ': logZ,
                            'CO_ratio': CO_ratio,
                            'log_scatt_factor': log_scatt_factor,
                            'scatt_slope': scatt_slope,
                            'log_cloudtop_P': log_cloudtop_P,
                            'Rs': Rs,
                            'error_multiple': error_multiple, 'T_star': T_star})

        if add_fit_params:
            fit_info.add_uniform_fit_param('R', 0.9*Rp, 1.1*Rp, 0, np.inf)
            fit_info.add_uniform_fit_param('T', 0.5*T, 1.5*T, 0, np.inf)
            fit_info.add_uniform_fit_param('logZ', -1, 3, -1, 3)
            fit_info.add_uniform_fit_param('CO_ratio', 0.2, 1.5, 0.2, 2.0)
            fit_info.add_uniform_fit_param('log_cloudtop_P', -1, 4, -np.inf, np.inf)
            fit_info.add_uniform_fit_param('log_scatt_factor', 0, 1, 0, 3)
            fit_info.add_uniform_fit_param('scatt_slope', 0, 8, 0, 8)
            fit_info.add_uniform_fit_param('error_multiple', 0.1, 10, 0, np.inf)
        return fit_info
