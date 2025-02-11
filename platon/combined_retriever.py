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
import sys

from .psis import psisloo
from .transit_depth_calculator import TransitDepthCalculator
from .eclipse_depth_calculator import EclipseDepthCalculator
from .fit_info import FitInfo

from .constants import METRES_TO_UM, M_jup, R_jup, R_earth, M_earth, R_sun
from ._params import _UniformParam
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .TP_profile import Profile
from .retrieval_result import RetrievalResult
from .custom_dynesty_result import CustomDynestyResult

class CombinedRetriever:
    def pretty_print(self, fit_info):
        if not hasattr(self, "last_lnprob"):
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

            if name in ["offset_transit", "offset_eclipse"]:
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
                    "Value {} for {} not between low and high limits {}-{}".format(
                        this_param.best_guess, name, this_param.low_lim, this_param.high_lim))
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

    @staticmethod
    def convert_clr_to_vmr(clrs):
        clr_bkg = -np.sum(clrs)
        clrs_with_bkg = np.append(clrs, clr_bkg)
        geometric_mean = 1 / np.sum(np.exp(clrs_with_bkg))
        vmrs_with_bkg = np.exp(clrs_with_bkg + np.log(geometric_mean))
        assert(np.around(np.sum(vmrs_with_bkg), decimals=5) == 1)
        return vmrs_with_bkg
    
    def _ln_like(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, ret_best_fit=False,
                 lnlike_per_point=False,
                 zero_opacities=[]):

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
        part_size = 10.**params_dict["log_part_size"]
        P_quench = 10.** params_dict["log_P_quench"]
        CH4_mult = 10.**params_dict["log_CH4_mult"]

        if params_dict["fit_vmr"]:
            assert(logZ is None and CO_ratio is None)
            gases = fit_info.gases
            vmrs = [10.**params_dict[f'log_{gas}'] for gas in gases[:-1]]
            vmrs.append(1 - np.sum(vmrs))
            if vmrs[-1] < 0: return -np.inf
        elif params_dict["fit_clr"]:
            assert(logZ is None and CO_ratio is None)
            gases = fit_info.gases
            clrs = [params_dict[f'clr_{gas}'] for gas in gases[:-1]]
            vmrs = self.convert_clr_to_vmr(clrs)
            if np.min(vmrs) < fit_info.clr_low_lim: return -np.inf
        else:
            vmrs = None
            gases = None
        if "n" in params_dict and params_dict["n"] is not None and "log_k" in params_dict:
            ri = params_dict["n"] - 1j * 10**params_dict["log_k"]
        else:
            ri = None
            
        if Rs <= 0 or Mp <= 0:
            return -np.inf

        ln_likelihood = np.array([])
        calculated_transit_depths = None
        transit_info_dict = None
        calculated_eclipse_depths = None
        eclipse_info_dict = None
        
        try:
            if measured_transit_depths is not None:
                if T is None:
                    raise ValueError("Must fit for T if using transit depths")

                transit_wavelengths, calculated_transit_depths, transit_info_dict = transit_calc.compute_depths(
                    Rs, Mp, Rp, T, logZ, CO_ratio, CH4_mult, gases, vmrs,
                    custom_abundances=None,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P, T_star=T_star,
                    T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size=part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)

                calculated_transit_depths[params_dict["offset_start"] : params_dict["offset_end"]] += params_dict["offset_transit"]
                residuals = calculated_transit_depths - measured_transit_depths
                scaled_errors = error_multiple * measured_transit_errors
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))
                
            if measured_eclipse_depths is not None:
                t_p_profile = Profile()
                t_p_profile.set_from_params_dict(params_dict["profile_type"], params_dict)

                if np.any(np.isnan(t_p_profile.temperatures)):
                    raise AtmosphereError("Invalid T/P profile")
                
                eclipse_wavelengths, calculated_eclipse_depths, eclipse_info_dict = eclipse_calc.compute_depths(
                    t_p_profile, Rs, Mp, Rp, T_star, logZ, CO_ratio, CH4_mult, gases, vmrs,
                    custom_abundances=None,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P,
                    T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size = part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)
                calculated_eclipse_depths[params_dict["offset_start"] : params_dict["offset_end"]] += params_dict["offset_eclipse"]
                residuals = calculated_eclipse_depths - measured_eclipse_depths
                scaled_errors = error_multiple * measured_eclipse_errors
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))

        except AtmosphereError as e:
            return -np.inf
        
        self.last_params = params
        self.last_lnprob = fit_info._ln_prior(params) + ln_likelihood.sum()
        
        if ret_best_fit:
            return calculated_transit_depths, transit_info_dict, calculated_eclipse_depths, eclipse_info_dict

        if lnlike_per_point:
            self.params_to_lnlike[tuple(params)] = ln_likelihood
            return ln_likelihood

        return ln_likelihood.sum()


    def _ln_prob(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, zero_opacities=[]):
        
        lnlike_per_point = self._ln_like(params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                                measured_transit_errors, measured_eclipse_depths,
                                measured_eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
        
        if not np.isscalar(lnlike_per_point):
            ln_like = lnlike_per_point.sum()
        else:
            assert(lnlike_per_point == -np.inf)
            ln_like = -np.inf

        return fit_info._ln_prior(params) + ln_like


    def run_emcee(self, transit_bins, transit_depths, transit_errors,
                  eclipse_bins, eclipse_depths, eclipse_errors,
                  fit_info, nwalkers=50,
                  nsteps=1000, include_condensation=True,
                  rad_method="xsec",
                  num_final_samples=100, zero_opacities=[]):
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
        rad_method : string, optional
            "xsec" for opacity sampling, "ktables" for correlated k
        zero_opacities : list of strings
            List of molecules to zero opacities for

        Returns
        -------
        result : RetrievalResult object
        '''
        self.params_to_lnlike = {}
        initial_positions = fit_info._generate_rand_param_arrays(nwalkers)
        transit_calc = None
        eclipse_calc = None

        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)       

        sampler = emcee.EnsembleSampler(
            nwalkers, fit_info._get_num_fit_params(), self._ln_prob,
            args=(transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                 eclipse_depths, eclipse_errors, zero_opacities))

        for i, result in enumerate(sampler.sample(
                initial_positions, iterations=nsteps)):
            if (i + 1) % 10 == 0:
                print("Step {}: {}".format(i + 1, self.pretty_print(fit_info)))

        best_params_arr = sampler.flatchain[np.argmax(
            sampler.flatlnprobability)]
        
        divisors, new_labels = self._get_divisors_labels(
            np.median(sampler.flatchain, axis=0),
            fit_info.fit_param_names)
        
        write_param_estimates_file(
            sampler.flatchain / divisors,
            best_params_arr / divisors,
            np.max(sampler.flatlnprobability),
            new_labels)

        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)
        retrieval_result = RetrievalResult(
            {"best_fit_params": best_params_arr,
                "acceptance_fraction": sampler.acceptance_fraction,
             "chain": sampler.chain,
             "flatchain": sampler.flatchain,
             "lnprobability": sampler.lnprobability,
             "flatlnprobability": sampler.flatlnprobability},             
            "emcee", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)
        equal_samples = np.copy(sampler.flatchain)
        np.random.shuffle(equal_samples)
        retrieval_result.random_transit_depths = []
        retrieval_result.random_eclipse_depths = []
        retrieval_result.random_TP_profiles = []        
        retrieval_result.pointwise_lnlikes = []
        for params in equal_samples[:num_final_samples]:
            ret = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, ret_best_fit=True)
            if ret == -np.inf: continue
            _, transit_info, _, eclipse_info = ret
                
            if transit_depths is not None:
                retrieval_result.random_transit_depths.append(transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            retrieval_result.pointwise_lnlikes.append(self.params_to_lnlike[tuple(params)])
        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
        return retrieval_result

    def _get_divisors_labels(self, medians, labels):
        divisors = np.ones(len(labels))
        new_labels = np.copy(labels)
        
        for i, l in enumerate(labels):            
            if l == "Rs":
                divisors[i] = R_sun
                new_labels[i] = "R_star/R_sun"
            if l == "Rp":
                if medians[i] > 0.5 * R_jup:
                    divisors[i] = R_jup
                    new_labels[i] = "R_p/R_j"
                else:
                    divisors[i] = R_earth
                    new_labels[i] = "R_p/R_e"
            if l == "Mp":
                if medians[i] > 0.1 * M_jup:
                    divisors[i] = M_jup
                    new_labels[i] = "M_p/M_j"
                else:
                    divisors[i] = M_earth
                    new_labels[i] = "M_p/M_e"
                    
        return divisors, new_labels
    
    def run_dynesty(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      maxiter=None, maxcall=None, nlive=250,
                      num_final_samples=100, zero_opacities=[],
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
        rad_method : string, optional
            "xsec" for opacity sampling, "ktables" for correlated k       
        nlive : int
            Number of live points to use for nested sampling
        zero_opacities : list of strings                                                                                                                                                                   
            List of molecules to zero opacities for
        **dynesty_kwargs : keyword arguments to pass to dynesty's NestedSampler

        Returns
        -------
        result : RetrievalResult object
        '''        
        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        def dynesty_ln_like(cube):
            lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                    eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
            if not np.isscalar(lnlike_per_point):
                ln_like = lnlike_per_point.sum()
            else:
                assert(lnlike_per_point == -np.inf)
                ln_like = -np.inf
            
            if np.random.randint(100) == 0:
                print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
            return ln_like

        num_dim = fit_info._get_num_fit_params()
        sampler = NestedSampler(dynesty_ln_like, transform_prior, num_dim, bound='multi', nlive=nlive, **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall)
        result = CustomDynestyResult(sampler.results)
        result.logp = result.logl + np.array([fit_info._ln_prior(params) for params in result.samples])
        best_params_arr = result.samples[np.argmax(result.logp)]

        normalized_weights = np.exp(result.logwt - np.max(result.logwt))
        normalized_weights /= np.sum(normalized_weights)
        result.weights = normalized_weights                                
        equal_samples = dynesty.utils.resample_equal(result.samples, result.weights)
        np.random.shuffle(equal_samples)

        divisors, new_labels = self._get_divisors_labels(
            np.median(equal_samples, axis=0),
            fit_info.fit_param_names)
        
        write_param_estimates_file(
            equal_samples / divisors,
            best_params_arr / divisors,
            np.max(result.logp),
            new_labels)
        
        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)

        retrieval_result = RetrievalResult(
            result, "dynesty", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)

        retrieval_result.random_transit_depths = []
        retrieval_result.random_eclipse_depths = []
        retrieval_result.random_TP_profiles = []
        retrieval_result.pointwise_lnlikes = []
        for params in equal_samples[:num_final_samples]:
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, ret_best_fit=True)
            if transit_depths is not None:
                retrieval_result.random_transit_depths.append(transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            retrieval_result.pointwise_lnlikes.append(self.params_to_lnlike[tuple(params)])

        #Calculate LOO-CV scores
        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
                    
        return retrieval_result


    def run_multinest(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      maxiter=None, maxcall=None, nlive=250,
                      num_final_samples=100, zero_opacities=[],
                      **dynesty_kwargs):
        import pymultinest
        
        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        def multinest_ln_like(cube):
            lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                    eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
            if not np.isscalar(lnlike_per_point):
                ln_like = lnlike_per_point.sum()
            else:
                assert(lnlike_per_point == -np.inf)
                ln_like = -np.inf
            
            if np.random.randint(100) == 0:
                print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
            return ln_like

        num_dim = fit_info._get_num_fit_params()
        basename = "multinest_" + str(np.random.randint(1000))
        result = pymultinest.solve(LogLikelihood=multinest_ln_like, Prior=transform_prior,
                                   n_dims=num_dim, outputfiles_basename=basename, verbose=True, resume=False, n_live_points=nlive)
        a = pymultinest.Analyzer(outputfiles_basename=basename, n_params=num_dim)
        data = a.get_data()
        result["samples"] = data[:,2:]
        result["logp"] = np.log(data[:,0])
        result["logl"] = -0.5 * data[:,1]
        best_params_arr = result["samples"][np.argmax(result["logp"])]
        
        equal_samples = a.get_equal_weighted_posterior()[:,:-1]
        np.random.shuffle(equal_samples)
        result["equal_samples"] = equal_samples
        
        divisors, new_labels = self._get_divisors_labels(
            np.median(equal_samples, axis=0),
            fit_info.fit_param_names)
        
        write_param_estimates_file(
            equal_samples / divisors,
            best_params_arr / divisors,
            np.max(result["logp"]),
            new_labels)

        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)

        retrieval_result = RetrievalResult(
            result, "pymultinest", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)

        retrieval_result.random_transit_depths = []
        retrieval_result.random_eclipse_depths = []
        retrieval_result.random_TP_profiles = []
        retrieval_result.pointwise_lnlikes = []
        for params in equal_samples[:num_final_samples]:
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, ret_best_fit=True)
            if transit_depths is not None:                                                
                retrieval_result.random_transit_depths.append(transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            retrieval_result.pointwise_lnlikes.append(self.params_to_lnlike[tuple(params)])

        #Calculate LOO-CV scores
        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
        return retrieval_result
        

    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T=None, logZ=0, CO_ratio=0.53, log_CH4_mult=0,
                             free_retrieval=False,
                             log_cloudtop_P=np.inf, log_scatt_factor=0,
                             scatt_slope=4, error_multiple=1, T_star=None,
                             T_spot=None, spot_cov_frac=None,
                             frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size=-6,
                             n=None, log_k=-np.inf,
                             log_P_quench=-99,
                             offset_transit=0, offset_eclipse=0, offset_start=0, offset_end=sys.maxsize,
                             fit_vmr=False, fit_clr=False,
                             profile_type = 'isothermal', **profile_kwargs):
        '''Get a :class:`.FitInfo` object filled with best guess values.  A few
        parameters are required, but others can be set to default values if you
        do not want to specify them.  All parameters are in SI.  For 
        information on the parameters not described below, see the documentation
        for :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths` and :func:`~platon.eclipse_depth_calculator.EclipseDepthCalculator.compute_depths`

        Parameters
        ----------
        n : float
            Real component of the refractive index of haze particles. Set to
            None to disable Mie scattering
        log_k : float
            log10 of the imaginary component of the refractive index of haze
            particles.  Set to -np.inf for k=0
        offset_transit : float
            Offset of transit data, identified by indexes offset_start and offset_end (e.g. obs[offset_start:offset_end]).
            A positive offset means the observed transit depths are decreased before comparing to the model.
        offset_eclipse : float
            Same as above, but for eclipse depths.
        profile_type : string
            "isothermal", "parametric" (Madhusudhan & Seager 2009) or 
            "radiative_solution" (Line et al 2013) T/P profile 
            parameterizations.  This profile applies to the dayside only,
            and hence is only relevant for eclipse depths.
        profile_kwargs : kwargs
            T/P profile arguments.  For "isothermal": T_day.  For "parametric":
            T0, P1, alpha1, alpha2, P3, T3.  For "radiative_solution":
            T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, log_gamma2,
            alpha, and T_int (optional).  We recommend that T_star, Rs, a, and 
            Mp be fixed, and that T_int be omitted (which sets it to 100 K).
            

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
