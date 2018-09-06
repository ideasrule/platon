from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import nestle
import copy

from .transit_depth_calculator import TransitDepthCalculator
from .fit_info import FitInfo
from .constants import METRES_TO_UM
from ._params import _UniformParam
from .errors import AtmosphereError
from .save_best_fit import write_param_estimates_file


class Retriever:
    def _validate_params(self, fit_info, calculator):
        # This assumes that the valid parameter space is rectangular, so that
        # the bounds for each parameter can be treated separately. Unfortunately
        # there is no good way to validate Gaussian parameters, which have
        # infinite range.
        fit_info = copy.deepcopy(fit_info)
        
        if fit_info.all_params["ri"].best_guess is None:
            # Not using Mie scattering
            if fit_info.all_params["log_number_density"].best_guess != -np.inf:
                raise ValueError("log number density must be -inf if not using Mie scattering")            
        else:
            if fit_info.all_params["log_scatt_factor"].best_guess != -np.inf:
                raise ValueError("log scattering factor must be -inf if using Mie scattering")           
            
        
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
                    fit_info._get("T"),
                    fit_info._get("logZ"),
                    fit_info._get("CO_ratio"),
                    10**fit_info._get("log_cloudtop_P"))

    def _ln_prob(self, params, calculator, fit_info, measured_depths,
                 measured_errors, plot=False):

        if not fit_info._within_limits(params):
            return -np.inf

        params_dict = fit_info._interpret_param_array(params)
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
        T_spot = params_dict["T_spot"]
        spot_cov_frac = params_dict["spot_cov_frac"]
        frac_scale_height = params_dict["frac_scale_height"]
        number_density = 10.0**params_dict["log_number_density"]
        part_size = 10.0**params_dict["log_part_size"]
        ri = params_dict["ri"]

        if Rs <= 0 or Mp <= 0:
            return -np.inf

        try:
            wavelengths, calculated_depths = calculator.compute_depths(
                Rs, Mp, R, T, logZ, CO_ratio,
                scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                cloudtop_pressure=cloudtop_P, T_star=T_star,
                T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                frac_scale_height=frac_scale_height, number_density=number_density,
                part_size = part_size, ri = ri)
        except AtmosphereError as e:
            print(e)
            return -np.inf

        residuals = calculated_depths - measured_depths
        scaled_errors = error_multiple * measured_errors
        ln_prob = -0.5 * np.sum(residuals**2 / scaled_errors**2 + \
                                np.log(2 * np.pi * scaled_errors**2))

        if plot:
            plt.errorbar(
                METRES_TO_UM * wavelengths, measured_depths,
                yerr=measured_errors, fmt='.',color='k')
            plt.plot(METRES_TO_UM * wavelengths, calculated_depths, color='b')
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Transit depth")
            plt.xscale('log')
            plt.tight_layout()

        return fit_info._ln_prior(params) + ln_prob

    def run_emcee(self, wavelength_bins, depths, errors, fit_info, nwalkers=50,
                  nsteps=1000, include_condensation=True,
                  plot_best=False):
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
        include_condensation : bool, optional
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

        initial_positions = fit_info._generate_rand_param_arrays(nwalkers)
        calculator = TransitDepthCalculator(
            include_condensation=include_condensation)
        calculator.change_wavelength_bins(wavelength_bins)
        self._validate_params(fit_info, calculator)

        sampler = emcee.EnsembleSampler(
            nwalkers, fit_info._get_num_fit_params(), self._ln_prob,
            args=(calculator, fit_info, depths, errors))

        for i, result in enumerate(sampler.sample(
                initial_positions, iterations=nsteps)):
            if (i + 1) % 10 == 0:
                print(str(i + 1) + "/" + str(nsteps),
                      sampler.lnprobability[0, i], sampler.chain[0, i])

        best_params_arr = sampler.flatchain[np.argmax(
            sampler.flatlnprobability)]
        
        write_param_estimates_file(
            sampler.flatchain,
            best_params_arr,
            np.max(sampler.flatlnprobability),
            fit_info.fit_param_names)

        if plot_best:
            self._ln_prob(
                best_params_arr,
                calculator,
                fit_info,
                depths,
                errors,
                plot=True)
        return sampler

    def run_multinest(self, wavelength_bins, depths, errors, fit_info,
                      include_condensation=True, plot_best=False,
                      **nestle_kwargs):
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
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        plot_best : bool, optional
            If True, plots the best fit model with the data
        **nestle_kwargs : keyword arguments to pass to nestle's sample method

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
        calculator = TransitDepthCalculator(
            include_condensation=include_condensation)
        calculator.change_wavelength_bins(wavelength_bins)
        self._validate_params(fit_info, calculator)

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        def multinest_ln_prob(cube):
            return self._ln_prob(cube, calculator, fit_info, depths, errors)

        def callback(callback_info):
            print(callback_info["it"], callback_info["logz"],
                  transform_prior(callback_info["active_u"][0]))

        result = nestle.sample(
            multinest_ln_prob, transform_prior, fit_info._get_num_fit_params(),
            callback=callback, method='multi', **nestle_kwargs)

        best_params_arr = result.samples[np.argmax(result.logl)]
        
        write_param_estimates_file(
            nestle.resample_equal(result.samples, result.weights),
            best_params_arr,
            np.max(result.logl),
            fit_info.fit_param_names)

        if plot_best:
            self._ln_prob(best_params_arr, calculator, fit_info,
                          depths, errors, plot=True)
        return result

    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T, logZ=0, CO_ratio=0.53,
                             log_cloudtop_P=np.inf, log_scatt_factor=0,
                             scatt_slope=4, error_multiple=1, T_star=None,
                             T_spot=None, spot_cov_frac=None,frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size =-6, ri = None):
        '''Get a :class:`.FitInfo` object filled with best guess values.  A few
        parameters are required, but others can be set to default values if you
        do not want to specify them.  All parameters are in SI.

        Parameters
        ----------
        Rs : float
            Stellar radius
        Mp : float
            Planetary mass
        Rp : float
            Planetary radius
        T : float
            Temperature of the isothermal planetary atmosphere
        logZ : float
            Base-10 logarithm of the metallicity, in solar units
        CO_ratio : float, optional
            C/O atomic ratio in the atmosphere.  The solar value is 0.53.
        log_cloudtop_P : float, optional
            Base-10 log of the pressure level (in Pa) below which light cannot
            penetrate.  Use np.inf for a cloudless atmosphere.
        log_scatt_factor : float, optional
            Base-10 logarithm of scattering factoring, which make scattering
            that many times as strong. If `scatt_slope` is 4, corresponding to
            Rayleigh scattering, the absorption coefficients are simply
            multiplied by `scattering_factor`. If slope is not 4,
            `scattering_factor` is defined such that the absorption coefficient
            is that many times as strong as Rayleigh scattering at
            the reference wavelength of 1 um.
        scatt_slope : float, optional
            Wavelength dependence of scattering, with 4 being Rayleigh.
        error_multiple : float, optional
            All error bars are multiplied by this factor.
        T_star : float, optional
            Effective temperature of the star.  This is used to make wavelength
            binning of transit depths more accurate.
        T_spot : float, optional
            Effective temperature of the star spots. This is used to make
            wavelength dependent correction to the observed transit depths.
        spot_cov_frac : float, optional
            The spot covering fraction of the star by area. This is used to make
            wavelength dependent correction to the transit depths.

        Returns
        -------
        fit_info : :class:`.FitInfo` object
            This object is used to indicate which parameters to fit for, which
            to fix, and what values all parameters should take.'''

        fit_info = FitInfo({'Mp': Mp,
                            'R': Rp,
                            'T': T,
                            'logZ': logZ,
                            'CO_ratio': CO_ratio,
                            'log_scatt_factor': log_scatt_factor,
                            'scatt_slope': scatt_slope,
                            'log_cloudtop_P': log_cloudtop_P,
                            'Rs': Rs,
                            'error_multiple': error_multiple,
                            'T_star': T_star,
                            'T_spot': T_spot,
                            'spot_cov_frac': spot_cov_frac,
                            'frac_scale_height': frac_scale_height,
                            'log_number_density': log_number_density,
                            'log_part_size': log_part_size, 'ri': ri})
        return fit_info
