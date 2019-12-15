import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import dynesty.utils
import copy
import pickle

from .transit_depth_calculator import TransitDepthCalculator
from .eclipse_depth_calculator import EclipseDepthCalculator
from .fit_info import FitInfo
from .constants import METRES_TO_UM, M_jup, R_jup, R_sun
from ._params import _UniformParam
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .TP_profile import Profile

class CombinedRetriever:
    def pretty_print(self, fit_info):
        if self.last_lnprob is None:
            return
        
        line = "ln_prob={:.2e}\t".format(self.last_lnprob)
        for i, name in enumerate(fit_info.fit_param_names):            
            value = self.last_params[i]
            unit = ""
            if name == "Rs":
                value /= R_sun
                unit = "R_sun"
            if name == "Mp":
                value /= M_jup
                unit = "M_jup"
            if name == "Rp":
                value /= R_jup
                unit = "R_jup"
            if name == "T":
                unit = "K"

            if name == "T":
                format_str = "{:4.0f}"                
            elif abs(value) < 1e4: format_str = "{:.2f}"
            else: format_str = "{:.2e}"

            if name == "wfc3_offset_transit" or name == "wfc3_offset_eclipse":
                unit = "ppm"
                value *= 1e6
            
            format_str = "{}=" + format_str + " " + unit + "\t"
            line += format_str.format(name, value)
            
        return line
    
    def _validate_params(self, fit_info, calculator):
        # This assumes that the valid parameter space is rectangular, so that
        # the bounds for each parameter can be treated separately. Unfortunately
        # there is no good way to validate Gaussian parameters, which have
        # infinite range.
        fit_info = copy.deepcopy(fit_info)
        
        if fit_info.all_params["log_k"].best_guess is None:
            # Not using Mie scattering
            if fit_info.all_params["log_number_density"].best_guess != -np.inf:
                raise ValueError("log number density must be -inf if not using Mie scattering")            
        else:
            if fit_info.all_params["log_scatt_factor"].best_guess != 0:
                raise ValueError("log scattering factor must be 0 if using Mie scattering")           
            
        
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
                    None,
                    fit_info._get("logZ"),
                    fit_info._get("CO_ratio"),
                    10**fit_info._get("log_cloudtop_P"))

    def _ln_like(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, plot=False, wfc3_start=1e-6, wfc3_end=1.7e-6):

        if not fit_info._within_limits(params):
            return -np.inf

        params_dict = fit_info._interpret_param_array(params)
        
        Rp = params_dict["Rp"]
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
        P_quench = 10 ** params_dict["log_P_quench"]
        wfc3_offset_transit = params_dict["wfc3_offset_transit"]
        wfc3_offset_eclipse = params_dict["wfc3_offset_eclipse"]
        
        if "n" in params_dict and "log_k" in params_dict:
            ri = params_dict["n"] - 1j * 10**params_dict["log_k"]
        else:
            ri = None

        if Rs <= 0 or Mp <= 0:
            return -np.inf

        ln_likelihood = 0
        try:
            if measured_transit_depths is not None:
                if T is None:
                    raise ValueError("Must fit for T if using transit depths")
                
                transit_wavelengths, calculated_transit_depths, info_dict = transit_calc.compute_depths(
                    Rs, Mp, Rp, T, logZ, CO_ratio,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P, T_star=T_star,
                    T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size=part_size, ri=ri, P_quench=P_quench, full_output=True)
                calculated_transit_depths[np.logical_and(transit_wavelengths >= wfc3_start, transit_wavelengths <= wfc3_end)] += wfc3_offset_transit
                residuals = calculated_transit_depths - measured_transit_depths
                scaled_errors = error_multiple * measured_transit_errors
                ln_likelihood += -0.5 * np.sum(residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2))
                
                if plot:
                    chi_sqr = np.sum(residuals**2 / measured_transit_errors**2)
                    print("Transit chi_sqr:", chi_sqr)
                    plt.figure(1)
                    plt.plot(METRES_TO_UM * info_dict["unbinned_wavelengths"],
                             info_dict["unbinned_depths"],
                             alpha=0.2, color='b', label="Calculated (unbinned)")
                    plt.errorbar(METRES_TO_UM * transit_wavelengths, measured_transit_depths,
                                 yerr = measured_transit_errors, fmt='.', color='k', label="Observed")
                    plt.scatter(METRES_TO_UM * transit_wavelengths,
                                calculated_transit_depths,
                                color='r', label="Calculated (binned)")
                    plt.xlabel("Wavelength ($\mu m$)")
                    plt.ylabel("Transit depth")
                    plt.xscale('log')
                    plt.tight_layout()
                    plt.legend()
                    plt.savefig("best_fit_transit.pdf")

            if measured_eclipse_depths is not None:
                t_p_profile = Profile()
                t_p_profile.set_from_params_dict(params_dict["profile_type"], params_dict)

                if np.any(np.isnan(t_p_profile.temperatures)):
                    raise AtmosphereError("Invalid T/P profile")
                
                eclipse_wavelengths, calculated_eclipse_depths, info_dict = eclipse_calc.compute_depths(
                    t_p_profile, Rs, Mp, Rp, T_star, logZ, CO_ratio,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P,
                    T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size = part_size, ri=ri, P_quench=P_quench, full_output=True)
                calculated_eclipse_depths[np.logical_and(eclipse_wavelengths >= wfc3_start, eclipse_wavelengths <= wfc3_end)] += wfc3_offset_eclipse
                residuals = calculated_eclipse_depths - measured_eclipse_depths
                scaled_errors = error_multiple * measured_eclipse_errors
                ln_likelihood += -0.5 * np.sum(residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2))
                
                if plot:
                    chi_sqr = np.sum(residuals**2 / measured_eclipse_depths**2)
                    print("Eclipse chi_sqr:", chi_sqr)
                    plt.figure(2)
                    plt.plot(METRES_TO_UM * info_dict["unbinned_wavelengths"],
                             info_dict["unbinned_eclipse_depths"],
                             alpha=0.2, color='b', label="Calculated (unbinned)")
                    plt.errorbar(METRES_TO_UM * eclipse_wavelengths,
                                 measured_eclipse_depths,
                                 yerr=measured_eclipse_errors, fmt='.', color='k', label="Observed")
                    plt.scatter(METRES_TO_UM * eclipse_wavelengths,
                                calculated_eclipse_depths,
                                color='r', label="Calculated (binned)")
                    plt.legend()
                    plt.xlabel("Wavelength ($\mu m$)")
                    plt.ylabel("Eclipse depth")
                    plt.xscale('log')
                    plt.tight_layout()
                    plt.legend()
                    plt.savefig("best_fit_eclipse.pdf")
                
        except AtmosphereError as e:
            print(e)
            return -np.inf
        
        self.last_params = params
        self.last_lnprob = fit_info._ln_prior(params) + ln_likelihood
        return ln_likelihood


    def _ln_prob(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, plot=False):
        
        ln_like = self._ln_like(params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                                measured_transit_errors, measured_eclipse_depths,
                                measured_eclipse_errors, plot=plot)
        return fit_info._ln_prior(params) + ln_like

    
    
    def run_emcee(self, transit_bins, transit_depths, transit_errors,
                  eclipse_bins, eclipse_depths, eclipse_errors,
                  fit_info, nwalkers=50,
                  nsteps=1000, include_condensation=True,
                  plot_best=False):
        '''Runs affine-invariant MCMC to retrieve atmospheric parameters.

        Parameters
        ----------
        transit_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        transit_depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        transit_errors : array_like, length N
            Errors on the aforementioned transit depths
        eclipse_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        eclipse_depths : array_like, length N
            Measured eclipse depths for the specified wavelength bins
        eclipse_errors : array_like, length N
            Errors on the aforementioned eclipse depths
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
        transit_calc = TransitDepthCalculator(
            include_condensation=include_condensation)
        transit_calc.change_wavelength_bins(transit_bins)
        eclipse_calc = EclipseDepthCalculator(
            include_condensation=include_condensation)
        eclipse_calc.change_wavelength_bins(eclipse_bins)
       
        self._validate_params(fit_info, transit_calc)

        sampler = emcee.EnsembleSampler(
            nwalkers, fit_info._get_num_fit_params(), self._ln_prob,
            args=(transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                 eclipse_depths, eclipse_errors))

        for i, result in enumerate(sampler.sample(
                initial_positions, iterations=nsteps)):
            if (i + 1) % 10 == 0:
                print("Step {}: {}".format(i + 1, self.pretty_print(fit_info)))

        best_params_arr = sampler.flatchain[np.argmax(
            sampler.flatlnprobability)]
        
        write_param_estimates_file(
            sampler.flatchain,
            best_params_arr,
            np.max(sampler.flatlnprobability),
            fit_info.fit_param_names)

        if plot_best:
             self._ln_prob(best_params_arr, transit_calc, eclipse_calc, fit_info,
                          transit_depths, transit_errors,
                          eclipse_depths, eclipse_errors, plot=True)

        return sampler

    def run_multinest(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, plot_best=False,
                      maxiter=None, maxcall=None, nlive=100,
                      **dynesty_kwargs):
        '''Runs nested sampling to retrieve atmospheric parameters.

        Parameters
        ----------
        transit_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        transit_depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        transit_errors : array_like, length N
            Errors on the aforementioned transit depths
        eclipse_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        eclipse_depths : array_like, length N
            Measured eclipse depths for the specified wavelength bins
        eclipse_errors : array_like, length N
            Errors on the aforementioned eclipse depths
        fit_info : :class:`.FitInfo` object
            Tells us what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        plot_best : bool, optional
            If True, plots the best fit model with the data
        nlive : int
            Number of live points to use for nested sampling
        **dynesty_kwargs : keyword arguments to pass to dynesty's NestedSampler

        Returns
        -------
        result : Result object
            This returns dynesty's NestedSampler 'results' field, slightly
            modified.  The object is
            dictionary-like and has many useful items.  For example,
            result.samples (or alternatively, result["samples"]) are the
            parameter values of each sample, result.logwt contains the
            log(weights), result.weights contains the normalized weights 
            (this is added by PLATON), 
            result.logl contains the ln likelihoods, and result.logp
            contains the ln posteriors (this is added by PLATON).  result.logz
            is the natural logarithm of the evidence.
        '''
        transit_calc = TransitDepthCalculator(
            include_condensation=include_condensation)
        transit_calc.change_wavelength_bins(transit_bins)
        eclipse_calc = EclipseDepthCalculator()
        eclipse_calc.change_wavelength_bins(eclipse_bins)
        
        self._validate_params(fit_info, transit_calc)

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        def multinest_ln_like(cube):
            ln_like = self._ln_like(cube, transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                 eclipse_depths, eclipse_errors)
            if np.random.randint(100) == 0:
                print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
            return ln_like

        num_dim = fit_info._get_num_fit_params()
        sampler = NestedSampler(multinest_ln_like, transform_prior, num_dim, bound='multi',
                                update_interval=float(num_dim), nlive=nlive, **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall)
        result = sampler.results
        
        result.logp = result.logl + np.array([fit_info._ln_prior(params) for params in result.samples])
        best_params_arr = result.samples[np.argmax(result.logp)]

        normalized_weights = np.exp(result.logwt - np.max(result.logwt))
        normalized_weights /= np.sum(normalized_weights)
        result.weights = normalized_weights

        with open("dynesty_result.pkl", "wb") as f:
            pickle.dump(result, f)
        
        write_param_estimates_file(
            dynesty.utils.resample_equal(result.samples, normalized_weights),
            best_params_arr,
            np.max(result.logp),
            fit_info.fit_param_names)

        if plot_best:
            self._ln_prob(best_params_arr, transit_calc, eclipse_calc, fit_info,
                          transit_depths, transit_errors,
                          eclipse_depths, eclipse_errors, plot=True)

            plt.figure(3)
            dyplot.runplot(result)
            plt.savefig("dyplot_runplot.png")
            plt.figure(4)
            dyplot.traceplot(result)
            plt.savefig("dyplot_traceplot.png")
        return result

    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T=None, logZ=0, CO_ratio=0.53,
                             log_cloudtop_P=np.inf, log_scatt_factor=0,
                             scatt_slope=4, error_multiple=1, T_star=None,
                             T_spot=None, spot_cov_frac=None,
                             frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size=-6,
                             n=None, log_k=None,
                             log_P_quench=-99,
                             profile_type = 'isothermal', **profile_kwargs):
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
        all_variables = locals().copy()
        del all_variables["profile_kwargs"]
        all_variables.update(profile_kwargs)
        
        fit_info = FitInfo(all_variables)
        return fit_info
