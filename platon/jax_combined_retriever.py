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
import time

from .psis import psisloo
from .jax_transit_depth_calculator import TransitDepthCalculator
from .eclipse_depth_calculator import EclipseDepthCalculator
from .fit_info import FitInfo

from .constants import METRES_TO_UM, M_jup, R_jup, R_earth, M_earth, R_sun
from ._params import _UniformParam, _GaussianParam, _Param
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .TP_profile import Profile
from .retrieval_result import RetrievalResult
from .custom_dynesty_result import CustomDynestyResult

class _JaxLikelihoodSetup:
    """Container for the JIT-compiled JAX likelihood and its metadata."""

    param_names = None
    default_arr = None
    all_param_defaults = None
    transit_data = None
    eclipse_data = None
    transit_forward = None
    eclipse_forward = None
    per_point_lnlike = None
    n_transit_points = 0
    n_eclipse_points = 0

    def params_dict_from_array(self, params_arr):
        params_dict = dict(self.all_param_defaults)
        for i, name in enumerate(self.param_names):
            params_dict[name] = float(params_arr[i])
        return params_dict

    def best_fit_transit_depths(self, params_arr):
        """Binned transit depths from the JAX model, or None if invalid."""
        if self.transit_data is None:
            return None
        from ._jax_forward_model import validate_runtime_params_jax
        params_dict = self.params_dict_from_array(
            np.asarray(params_arr, dtype=float))
        if not bool(np.asarray(
                validate_runtime_params_jax(params_dict, self.transit_data))):
            return None
        return np.asarray(
            self.transit_forward(params_dict, self.transit_data), dtype=float)

    def best_fit_eclipse_depths(self, params_arr):
        """Binned eclipse depths from the JAX model, or None if invalid."""
        if self.eclipse_data is None:
            return None
        from ._jax_forward_model import validate_runtime_params_jax
        params_dict = self.params_dict_from_array(
            np.asarray(params_arr, dtype=float))
        if not bool(np.asarray(
                validate_runtime_params_jax(params_dict, self.eclipse_data))):
            return None
        return np.asarray(
            self.eclipse_forward(params_dict, self.eclipse_data), dtype=float)


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

            if name in ["offset_niriss", "offset_nrs1", "offset_miri"]:
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

        if "error_additive" in fit_info.fit_param_names:
            ea = fit_info.all_params["error_additive"]
            if isinstance(ea, _UniformParam):
                if ea.low_lim < 0 or ea.high_lim > 1e-3:
                    raise ValueError(
                        "error_additive range must be within [0, 1e-3], got [{}, {}]".format(
                            ea.low_lim, ea.high_lim))

            # The transit path is profile-based now, and non-isothermal
            # retrievals do not populate a scalar ``T``. Validity is enforced
            # during the forward-model evaluation, so keep the prior-rectangle
            # checks above but avoid the stale scalar-temperature preflight.

    def add_sulfur_transition(self, fit_info,
                              log_H2S_vmr_range=(-8, -2),
                              log_CS2_vmr_range=(-8, -2)):
        """Add H2S/CS2 pressure-transition chemistry to fit_info.

        H2S is present at P > P_transition (deep atmosphere) with VMR
        10**log_H2S_vmr; CS2 at P < P_transition (upper atmosphere) with
        10**log_CS2_vmr.  The two VMRs and the transition pressure are all
        free parameters.  log10_P_S_transition (log10 in Pa) is initially
        added with a uniform prior [0, 2]; remove it from fit_param_names
        and call add_gaussian_fit_param to replace it with a Gaussian prior.
        """
        lo_h2s, hi_h2s = log_H2S_vmr_range
        lo_cs2, hi_cs2 = log_CS2_vmr_range
        fit_info.all_params["log_H2S_vmr"] = _Param(0.5 * (lo_h2s + hi_h2s))
        fit_info.add_uniform_fit_param("log_H2S_vmr", lo_h2s, hi_h2s)
        fit_info.all_params["log_CS2_vmr"] = _Param(0.5 * (lo_cs2 + hi_cs2))
        fit_info.add_uniform_fit_param("log_CS2_vmr", lo_cs2, hi_cs2)
        fit_info.all_params["log10_P_S_transition"] = _Param(1.0)
        fit_info.add_uniform_fit_param("log10_P_S_transition", 0.0, 2.0)

    @staticmethod
    def convert_clr_to_vmr(clrs):
        clr_bkg = -np.sum(clrs)
        clrs_with_bkg = np.append(clrs, clr_bkg)
        geometric_mean = 1 / np.sum(np.exp(clrs_with_bkg))
        vmrs_with_bkg = np.exp(clrs_with_bkg + np.log(geometric_mean))
        assert(np.around(np.sum(vmrs_with_bkg), decimals=5) == 1)
        return vmrs_with_bkg

    @staticmethod
    def _get_transit_offset_windows(params_dict, spectrum_length):
        windows = {}

        def add_window(offset_name, start, end):
            if offset_name not in params_dict:
                return
            try:
                i_start = int(start)
                i_end = int(end)
            except (TypeError, ValueError):
                return
            i_start = max(0, min(i_start, spectrum_length))
            i_end = max(0, min(i_end, spectrum_length))
            if i_end <= i_start:
                return
            windows[offset_name] = (i_start, i_end)

        explicit_windows = params_dict.get("transit_offset_windows")
        if isinstance(explicit_windows, dict):
            for offset_name, bounds in explicit_windows.items():
                if isinstance(bounds, dict):
                    add_window(offset_name, bounds.get("start"), bounds.get("end"))
                elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                    add_window(offset_name, bounds[0], bounds[1])

        for key in params_dict:
            if not key.startswith("offset_") or key.endswith("_start") or key.endswith("_end"):
                continue
            if key == "offset_eclipse":
                continue
            start_key = key + "_start"
            end_key = key + "_end"
            if start_key in params_dict and end_key in params_dict:
                add_window(key, params_dict[start_key], params_dict[end_key])

        if "offset_transit" in params_dict and "offset_start" in params_dict and "offset_end" in params_dict:
            add_window("offset_transit", params_dict["offset_start"], params_dict["offset_end"])

        return windows.items()

    @staticmethod
    def _has_dynamic_transit_offset_indices(fit_info):
        for name in fit_info.fit_param_names:
            if name.endswith("_start") or name.endswith("_end"):
                return True
        return False

    def _get_transit_offset_ops(self, fit_info, params_dict, spectrum_length):
        if self._has_dynamic_transit_offset_indices(fit_info):
            return tuple(self._get_transit_offset_windows(params_dict, spectrum_length))

        windows_obj = params_dict.get("transit_offset_windows")
        cache_key = (id(fit_info), spectrum_length, id(windows_obj))
        cache = getattr(self, "_transit_offset_ops_cache", None)
        if cache is not None and cache.get("key") == cache_key:
            return cache["ops"]

        ops = tuple(self._get_transit_offset_windows(params_dict, spectrum_length))
        self._transit_offset_ops_cache = {"key": cache_key, "ops": ops}
        return ops

    def _setup_jax_likelihood(self, fit_info, sampler_name,
                              transit_calc, transit_bins, transit_depths,
                              transit_errors,
                              eclipse_calc, eclipse_bins, eclipse_depths,
                              eclipse_errors, zero_opacities=[]):
        """Build a JIT-compiled per-point log-likelihood over transit and/or
        eclipse data.

        The returned object exposes ``per_point_lnlike(params_arr)``, which
        gives one log-likelihood term per data point, transit points first and
        then eclipse points — the same ordering as :meth:`_ln_like`, so the
        values stay compatible with the PSIS-LOO bookkeeping.

        Raises ``UnsupportedJAXFeatureError`` if any requested option has no
        JAX implementation; callers should catch that and fall back to the
        NumPy path.
        """
        import jax
        import jax.numpy as jnp
        from ._jax_forward_model import (
            prepare_jax_data, validate_runtime_params_jax,
            jax_compute_transit_depths, jax_compute_transit_depths_patchy,
            jax_compute_transit_depths_limb_asym)
        from ._jax_eclipse_model import (
            prepare_jax_eclipse_data, jax_compute_eclipse_depths)

        setup = _JaxLikelihoodSetup()
        setup.param_names = list(fit_info.fit_param_names)
        setup.default_arr = np.array(
            [fit_info.all_params[n].best_guess for n in setup.param_names])
        defaults_dict = fit_info._interpret_param_array(setup.default_arr)

        T_star_val = defaults_dict.get("T_star", None)
        T_spot_val = defaults_dict.get("T_spot", None)
        spot_cov_val = defaults_dict.get("spot_cov_frac", None)

        all_param_defaults = {}
        for key, val in defaults_dict.items():
            if val is None:
                continue  # e.g. T_quench when it is not being fitted
            if isinstance(val, (int, float, np.floating, np.integer)):
                all_param_defaults[key] = float(val)
            else:
                all_param_defaults[key] = val
        setup.all_param_defaults = all_param_defaults

        has_transit = transit_bins is not None
        has_eclipse = eclipse_bins is not None

        if has_transit:
            print(f"Preparing JAX transit forward model for {sampler_name}...")
            setup.transit_data = prepare_jax_data(
                transit_calc.atm, transit_calc.atm.abundance_getter,
                transit_bins, T_star_val, T_spot_val, spot_cov_val,
                False, fit_info, len(transit_depths),
                zero_opacities=zero_opacities)
            if setup.transit_data.get("limb_asym", False):
                setup.transit_forward = jax_compute_transit_depths_limb_asym
            elif setup.transit_data.get("patchy", False):
                setup.transit_forward = jax_compute_transit_depths_patchy
            else:
                setup.transit_forward = jax_compute_transit_depths
            setup.n_transit_points = len(transit_depths)
            jax_transit_measured = jnp.array(transit_depths, dtype=jnp.float32)
            jax_transit_errors = jnp.array(transit_errors, dtype=jnp.float32)

        if has_eclipse:
            print(f"Preparing JAX emission forward model for {sampler_name}...")
            setup.eclipse_data = prepare_jax_eclipse_data(
                eclipse_calc.atm, eclipse_calc.atm.abundance_getter,
                eclipse_bins, T_star_val, T_spot_val, spot_cov_val,
                False, fit_info, len(eclipse_depths),
                zero_opacities=zero_opacities)
            setup.eclipse_forward = jax_compute_eclipse_depths
            setup.n_eclipse_points = len(eclipse_depths)
            jax_eclipse_measured = jnp.array(eclipse_depths, dtype=jnp.float32)
            jax_eclipse_errors = jnp.array(eclipse_errors, dtype=jnp.float32)

        n_points = setup.n_transit_points + setup.n_eclipse_points
        validity_data = (setup.transit_data if has_transit
                         else setup.eclipse_data)
        param_names = setup.param_names
        fit_error_additive = "error_additive" in param_names

        def _gaussian_per_point(binned, measured, errors, params_dict,
                                allow_additive):
            if allow_additive and fit_error_additive:
                err_add = jnp.float32(params_dict["error_additive"])
                scaled_err = jnp.sqrt(errors ** 2 + err_add ** 2)
            else:
                err_mult = jnp.float32(params_dict["error_multiple"])
                scaled_err = err_mult * errors
            resid = binned - measured
            return -0.5 * (resid ** 2 / scaled_err ** 2
                           + jnp.log(jnp.float32(2 * jnp.pi) * scaled_err ** 2))

        @jax.jit
        def _jax_per_point_lnlike(params_arr):
            params_dict = dict(all_param_defaults)
            for i, name in enumerate(param_names):
                params_dict[name] = params_arr[i]
            is_valid = validate_runtime_params_jax(params_dict, validity_data)

            def invalid_branch(_):
                return jnp.full((n_points,), jnp.nan, dtype=jnp.float32)

            def valid_branch(_):
                parts = []
                if has_transit:
                    binned = setup.transit_forward(
                        params_dict, setup.transit_data)
                    parts.append(_gaussian_per_point(
                        binned, jax_transit_measured, jax_transit_errors,
                        params_dict, True))
                if has_eclipse:
                    binned = jax_compute_eclipse_depths(
                        params_dict, setup.eclipse_data)
                    # The NumPy path always scales eclipse errors by
                    # error_multiple, never by error_additive.
                    parts.append(_gaussian_per_point(
                        binned, jax_eclipse_measured, jax_eclipse_errors,
                        params_dict, False))
                return jnp.concatenate(parts)

            return jax.lax.cond(is_valid, valid_branch, invalid_branch,
                                operand=None)

        setup.per_point_lnlike = _jax_per_point_lnlike

        print("JIT-compiling JAX forward model (first call may be slow)...")
        compile_params = jnp.array(setup.default_arr, dtype=jnp.float32)
        _ = _jax_per_point_lnlike(compile_params).block_until_ready()
        print(f"JAX forward model ready for {sampler_name}")
        return setup

    def _ln_like(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, ret_best_fit=False,
                 lnlike_per_point=False,
                 zero_opacities=[]):

        if not fit_info._within_limits(params):
            if ret_best_fit:
                return None, None, None, None
            return -np.inf

        params_dict = fit_info._interpret_param_array(params)

        Rp = params_dict["Rp"]
        logZ = params_dict["logZ"]
        CO_ratio = params_dict["CO_ratio"]
        scatt_factor = 10.0**params_dict["log_scatt_factor"]
        scatt_slope = params_dict["scatt_slope"]
        scatt_factor_evening = 10.0**params_dict.get(
            "log_scatt_factor_evening", 0.0)
        scatt_slope_evening = params_dict.get("scatt_slope_evening", 4.0)
        cloudtop_P = 10.0**params_dict["log_cloudtop_P"]
        error_multiple = params_dict["error_multiple"]
        Rs = params_dict["Rs"]
        Mp = params_dict["Mp"]
        T_star = params_dict["T_star"]
        T_spot = params_dict["T_spot"]
        spot_cov_frac = params_dict["spot_cov_frac"]
        cloud_cov_frac = params_dict["cloud_cov_frac"]
        frac_scale_height = params_dict["frac_scale_height"]
        number_density = 10.0**params_dict["log_number_density"]
        part_size = 10.**params_dict["log_part_size"]
        P_quench = 10.** params_dict["log_P_quench"]
        T_quench = params_dict.get("T_quench")
        quench_species = params_dict.get("quench_species")
        CH4_mult = 10.**params_dict["log_CH4_mult"]
        log_SO2 = params_dict.get('log_SO2')
        log_CH4 = params_dict.get('log_CH4')
        log_TiO = params_dict.get('log_TiO')
        log_VO = params_dict.get('log_VO')
        log_CS2 = params_dict.get('log_CS2')
        log_S = params_dict.get('log_S')  # tied H2S+CS2 free retrieval (optional)
        add_H_minus_absorption = bool(params_dict.get("add_H_minus_absorption", False))
        limb_asym = bool(params_dict.get("limb_asym", False))
        delta_T = params_dict.get("delta_T", 0.0)
        f_limb = params_dict.get("f_limb", 0.5)
        cloudtop_P_evening = 10.0**params_dict.get("log_cloudtop_P_evening", np.inf)
        abunds = None
        if params_dict["fit_vmr"]:
            assert(logZ is None and CO_ratio is None)
            gases = list(fit_info.gases)
            vmrs = [10.**params_dict[f'log_{gas}'] for gas in gases[:-1]]

            # Expand tied "S" gas into H2S + CS2 for the non-JAX path
            # (_atmosphere_solver doesn't know about "S"). 10**log_S is
            # the elemental S abundance; atom-budget split gives equal
            # molecular VMRs of (1/3) * 10**log_S to each of H2S, CS2.
            # The (1/3) * 10**log_S not consumed by molecules falls into
            # the background, consistent with the JAX path's bg correction
            # via vmr_gas_total_frac.
            if "S" in gases[:-1]:
                if "H2S" in gases or "CS2" in gases:
                    raise ValueError(
                        "fit_info.gases contains tied 'S' alongside H2S or "
                        "CS2. Use 'S' OR fit H2S/CS2 separately, not both.")
                s_idx = gases.index("S")
                s_vmr = vmrs[s_idx]
                split_vmr = s_vmr / 3.0
                gases = gases[:s_idx] + ["H2S", "CS2"] + gases[s_idx + 1:]
                vmrs = vmrs[:s_idx] + [split_vmr, split_vmr] + vmrs[s_idx + 1:]

            # Sulfur transition override (non-JAX approximation: uniform VMR).
            log_H2S_vmr = params_dict.get("log_H2S_vmr")
            if log_H2S_vmr is not None:
                if "H2S" in gases[:-1]:
                    vmrs[gases.index("H2S")] = 10.0 ** log_H2S_vmr
                if "CS2" in gases[:-1]:
                    vmrs[gases.index("CS2")] = 10.0 ** params_dict["log_CS2_vmr"]

            vmrs.append(1 - np.sum(vmrs))
            if vmrs[-1] < 0: return -np.inf
        elif params_dict["fit_clr"]:
            assert(logZ is None and CO_ratio is None)
            gases = fit_info.gases
            if "S" in gases:
                raise NotImplementedError(
                    "Tied 'S' gas in CLR mode is not yet supported in the "
                    "non-JAX summary path. Use fit_vmr=True instead.")
            clrs = [params_dict[f'clr_{gas}'] for gas in gases[:-1]]
            vmrs = self.convert_clr_to_vmr(clrs)
            if np.min(vmrs) < fit_info.clr_low_lim: return -np.inf
        else:
            vmrs = None
            gases = None
            if transit_calc is not None and params_dict.get("log_CS2") is not None:
                abunds = transit_calc.atm.abundance_getter.get(logZ, CO_ratio)
                abunds["CH4"] *= CH4_mult
                abunds["CS2"] = np.full_like(abunds["H2O"], 10.**params_dict["log_CS2"])
        if "n" in params_dict and params_dict["n"] is not None and "log_k" in params_dict:
            ri = params_dict["n"] - 1j * 10**params_dict["log_k"]
        else:
            ri = None
            
        if Rs <= 0 or Mp <= 0:
            return -np.inf
        if not np.isfinite(error_multiple) or error_multiple <= 0:
            return -np.inf

        ln_likelihood = np.array([])
        calculated_transit_depths = None
        transit_info_dict = None
        calculated_eclipse_depths = None
        eclipse_info_dict = None

        try:
            if measured_transit_depths is not None:
                t_p_profile = Profile()
                t_p_profile.set_from_params_dict(params_dict["profile_type"], params_dict)

                if np.any(np.isnan(t_p_profile.temperatures)):
                    raise AtmosphereError("Invalid T/P profile")

                if limb_asym:
                    # 1.5D limb-asymmetric model. The evening limb is the
                    # base profile rigidly shifted by +delta_T (hotter
                    # dayside morning, in the usual convention). Chemistry,
                    # Rp, Mp, Rs, stellar spectrum are shared; the two
                    # sectors are linearly blended after each full depth
                    # has been computed (see transit_depth_calculator).
                    t_p_profile_evening = Profile()
                    evening_params = dict(params_dict)
                    profile_type = params_dict.get("profile_type", "isothermal")
                    if profile_type == "isothermal":
                        evening_params["T"] = params_dict["T"] + delta_T
                    elif profile_type == "parametric":
                        evening_params["T0"] = params_dict["T0"] + delta_T
                    elif profile_type == "twopoint":
                        evening_params["T_top"] = params_dict["T_top"] + delta_T
                        evening_params["T_bottom"] = params_dict["T_bottom"] + delta_T
                    else:
                        raise ValueError(
                            f"limb_asym not supported for profile_type='{profile_type}'")
                    t_p_profile_evening.set_from_params_dict(profile_type, evening_params)

                    if np.any(np.isnan(t_p_profile_evening.temperatures)):
                        raise AtmosphereError("Invalid evening T/P profile")

                    transit_wavelengths, calculated_transit_depths, transit_info_dict = transit_calc.compute_depths_limb_asym(
                        t_p_profile, t_p_profile_evening,
                        Rs, Mp, Rp,
                        f_limb=f_limb,
                        logZ=None, CO_ratio=None,
                        CH4_mult=CH4_mult, gases=gases, vmrs=vmrs,
                        custom_abundances=abunds,
                        add_H_minus_absorption=add_H_minus_absorption,
                        scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                        scattering_factor_evening=scatt_factor_evening,
                        scattering_slope_evening=scatt_slope_evening,
                        cloudtop_pressure=cloudtop_P,
                        cloudtop_pressure_evening=cloudtop_P_evening,
                        T_star=T_star,
                        T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                        frac_scale_height=frac_scale_height, number_density=number_density,
                        part_size=part_size, ri=ri, P_quench=P_quench,
                        T_quench=T_quench, quench_species=quench_species,
                        full_output=ret_best_fit, zero_opacities=zero_opacities)
                elif float(cloud_cov_frac) != 1.0:
                    transit_wavelengths, calculated_transit_depths, transit_info_dict = transit_calc.compute_depths_patchy(
                        t_p_profile, Rs, Mp, Rp,
                        cloud_cov_frac=cloud_cov_frac,
                        logZ=None, CO_ratio=None,
                        custom_abundances=abunds,
                        CH4_mult=CH4_mult, gases=gases, vmrs=vmrs,
                        add_H_minus_absorption=add_H_minus_absorption,
                        scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                        cloudtop_pressure=cloudtop_P, T_star=T_star,
                        T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                        frac_scale_height=frac_scale_height, number_density=number_density,
                        part_size=part_size, ri=ri, P_quench=P_quench,
                        T_quench=T_quench, quench_species=quench_species,
                        full_output=ret_best_fit, zero_opacities=zero_opacities)
                else:
                    transit_wavelengths, calculated_transit_depths, transit_info_dict = transit_calc.compute_depths(
                        t_p_profile, Rs, Mp, Rp, None, None, CH4_mult, gases, vmrs,
                        custom_abundances=abunds,
                        add_H_minus_absorption=add_H_minus_absorption,
                        scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                        cloudtop_pressure=cloudtop_P, T_star=T_star,
                        T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                        frac_scale_height=frac_scale_height, number_density=number_density,
                        part_size=part_size, ri=ri, P_quench=P_quench,
                        T_quench=T_quench, quench_species=quench_species,
                        full_output=ret_best_fit, zero_opacities=zero_opacities)

                for offset_name, (start, end) in self._get_transit_offset_ops(
                        fit_info, params_dict, len(calculated_transit_depths)):
                    offset_value = params_dict[offset_name]
                    if not np.isscalar(offset_value) or not np.isfinite(offset_value):
                        return -np.inf
                    calculated_transit_depths[start:end] += offset_value

                residuals = calculated_transit_depths - measured_transit_depths
                error_additive = params_dict.get("error_additive", 0.0)
                if "error_additive" in fit_info.fit_param_names:
                    scaled_errors = np.sqrt(measured_transit_errors**2 + error_additive**2)
                else:
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
                if ("cloud_cov_frac" in fit_info.fit_param_names
                        or float(cloud_cov_frac) != 1.0):
                    # Patchy clouds: blend with a clear column, matching
                    # _jax_eclipse_model.  Binning is linear in depth, so
                    # blending binned depths equals binning blended ones.
                    _, clear_eclipse_depths, clear_info_dict = eclipse_calc.compute_depths(
                        t_p_profile, Rs, Mp, Rp, T_star, logZ, CO_ratio, CH4_mult, gases, vmrs,
                        custom_abundances=None,
                        scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                        cloudtop_pressure=np.inf,
                        T_spot=T_spot, spot_cov_frac=spot_cov_frac,
                        frac_scale_height=frac_scale_height, number_density=number_density,
                        part_size = part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)
                    f_cloud = float(cloud_cov_frac)
                    if eclipse_info_dict is not None:
                        # The rest of the dict (contrib, P_profile, taus...)
                        # describes the cloudy column; the clear column's
                        # contribution function is kept alongside it.
                        cloudy_unbinned = eclipse_info_dict["unbinned_eclipse_depths"]
                        clear_unbinned = clear_info_dict["unbinned_eclipse_depths"]
                        eclipse_info_dict["cloud_cov_frac"] = f_cloud
                        eclipse_info_dict["unbinned_eclipse_depths_cloudy"] = cloudy_unbinned
                        eclipse_info_dict["unbinned_eclipse_depths_clear"] = clear_unbinned
                        eclipse_info_dict["unbinned_eclipse_depths"] = (
                            f_cloud * cloudy_unbinned + (1 - f_cloud) * clear_unbinned)
                        eclipse_info_dict["contrib_clear"] = clear_info_dict["contrib"]
                        eclipse_info_dict["P_profile_clear"] = clear_info_dict["P_profile"]
                    calculated_eclipse_depths = (
                        f_cloud * calculated_eclipse_depths
                        + (1 - f_cloud) * clear_eclipse_depths)
                offset_eclipse = params_dict.get("offset_eclipse")
                offset_start = params_dict.get("offset_start")
                offset_end = params_dict.get("offset_end")
                if (offset_eclipse is not None and offset_start is not None
                        and offset_end is not None):
                    if not np.isscalar(offset_eclipse) or not np.isfinite(offset_eclipse):
                        return -np.inf
                    calculated_eclipse_depths[int(offset_start):int(offset_end)] += offset_eclipse
                residuals = calculated_eclipse_depths - measured_eclipse_depths
                scaled_errors = error_multiple * measured_eclipse_errors
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))

        except (AtmosphereError, ValueError, FloatingPointError, OverflowError):
            if ret_best_fit:
                return None, None, None, None
            return -np.inf

        ln_likelihood_sum = ln_likelihood.sum()
        if not np.isfinite(ln_likelihood_sum):
            if ret_best_fit:
                return None, None, None, None
            return -np.inf
        
        self.last_params = params
        self.last_lnprob = fit_info._ln_prior(params) + ln_likelihood_sum
        
        if ret_best_fit:
            return calculated_transit_depths, transit_info_dict, calculated_eclipse_depths, eclipse_info_dict

        if lnlike_per_point:
            self.params_to_lnlike[tuple(params)] = ln_likelihood
            return ln_likelihood

        return ln_likelihood_sum


    def _ln_prob(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, zero_opacities=[]):
        
        ln_like = self._ln_like(params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                                measured_transit_errors, measured_eclipse_depths,
                                measured_eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=False)

        if not np.isfinite(ln_like) or ln_like < -1e30:
            ln_like = -1e30

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
        try:
            if len(retrieval_result.pointwise_lnlikes) > 1:
                retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
            else:
                retrieval_result.loo_total = retrieval_result.loos = retrieval_result.loo_ks = None
        except Exception as e:
            print(f"LOO-CV skipped: {e}")
            retrieval_result.loo_total = retrieval_result.loos = retrieval_result.loo_ks = None
        return retrieval_result

    def run_numpyro(self, transit_bins, transit_depths, transit_errors,
                    eclipse_bins, eclipse_depths, eclipse_errors,
                    fit_info, num_warmup=500, num_samples=500,
                    include_condensation=True, rad_method="xsec",
                    num_chains=1, num_final_samples=100, zero_opacities=[],
                    target_accept_prob=0.85, rng_seed=0,
                    inverse_mass_matrix=None):
        '''Runs Hamiltonian Monte Carlo (NUTS) via numpyro to retrieve
        atmospheric parameters.

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
        num_warmup : int, optional
            Number of warmup (burn-in) steps for NUTS
        num_samples : int, optional
            Number of posterior samples to draw
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        rad_method : string, optional
            "xsec" for opacity sampling, "ktables" for correlated k
        num_chains : int, optional
            Number of independent MCMC chains
        num_final_samples : int, optional
            Number of samples used for posterior predictive checks and LOO-CV
        zero_opacities : list of strings
            List of molecules to zero opacities for
        target_accept_prob : float, optional
            Target acceptance probability for NUTS adaptation
        rng_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        result : RetrievalResult object
        '''
        import jax
        # Run in float32 for ~6x faster GPU compute on RTX 4090 etc.
        # The forward model is numerically stable in float32 after rescaling
        # (Rayleigh scattering, collisional absorption, log-space n_dens).
        jax.config.update("jax_enable_x64", False)
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, HMC
        from numpyro.infer.reparam import LocScaleReparam

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

        n_params = fit_info._get_num_fit_params()

        # Decide whether to use JAX autodiff or finite-difference fallback.
        # Autodiff is used for transit-only retrievals without eclipse depths,
        # free-retrieval VMR/CLR modes, or Mie scattering.
        # Emission depths go through a hard argmin (the tau ~ 1 photosphere
        # radius), so they are not usefully differentiable; HMC stays on the
        # transit-only path.  Eclipse retrievals still get the JAX speedup via
        # run_dynesty / run_multinest / run_ultranest.
        use_autodiff = (eclipse_bins is None and
                        transit_bins is not None)

        if use_autodiff:
            from ._jax_forward_model import prepare_jax_data, jax_log_likelihood

            # Get stellar params from fit_info defaults
            param_names = list(fit_info.fit_param_names)
            default_arr = np.array([fit_info.all_params[n].best_guess
                                    for n in param_names])
            defaults_dict = fit_info._interpret_param_array(default_arr)
            T_star_val = defaults_dict.get("T_star", None)
            T_spot_val = defaults_dict.get("T_spot", None)
            spot_cov_val = defaults_dict.get("spot_cov_frac", None)

            print("Preparing JAX forward model (analytic gradients via autodiff)...")
            jax_data = prepare_jax_data(
                transit_calc.atm, transit_calc.atm.abundance_getter,
                transit_bins, T_star_val, T_spot_val, spot_cov_val,
                False, fit_info, len(transit_depths),
                zero_opacities=zero_opacities)

            # Build default values dict for ALL parameters (fixed + fitted)
            all_param_defaults = {}
            for key in defaults_dict:
                val = defaults_dict[key]
                if val is None:
                    continue  # skip None params (e.g. T_quench when not fitted)
                elif isinstance(val, (int, float, np.floating, np.integer)):
                    all_param_defaults[key] = float(val)
                else:
                    all_param_defaults[key] = val

            jax_measured = jnp.array(transit_depths, dtype=jnp.float32)
            jax_errors = jnp.array(transit_errors, dtype=jnp.float32)

            def jax_ln_like(params_arr):
                return jax_log_likelihood(
                    params_arr, param_names, all_param_defaults,
                    jax_data, jax_measured, jax_errors)

            print("JIT-compiling forward model + gradient (first call may be slow)...")

        else:
            # Finite-difference fallback (for eclipse depths, etc.)
            # Compute characteristic scales for finite-difference step sizes
            fd_scales = np.zeros(n_params, dtype=np.float64)
            for i, name in enumerate(fit_info.fit_param_names):
                param = fit_info.all_params[name]
                if isinstance(param, _UniformParam):
                    fd_scales[i] = param.high_lim - param.low_lim
                elif isinstance(param, _GaussianParam):
                    fd_scales[i] = param.std
                else:
                    fd_scales[i] = max(1.0, abs(param.best_guess))

            def numpy_ln_like_fd(params_np):
                params_np = np.asarray(params_np, dtype=np.float64)
                result = self._ln_like(
                    params_np, transit_calc, eclipse_calc, fit_info,
                    transit_depths, transit_errors,
                    eclipse_depths, eclipse_errors,
                    zero_opacities=zero_opacities)
                if result == -np.inf or (not np.isscalar(result) and not np.all(np.isfinite(result))):
                    return np.float64(-1e30)
                if not np.isscalar(result):
                    return np.float64(np.sum(result))
                return np.float64(result)

            def numpy_grad_ln_like(params_np):
                params_np = np.asarray(params_np, dtype=np.float64)
                f0 = numpy_ln_like_fd(params_np)
                grad = np.zeros(n_params, dtype=np.float64)
                for i in range(n_params):
                    h = fd_scales[i] * 1e-2
                    p_plus = params_np.copy()
                    p_minus = params_np.copy()
                    p_plus[i] += h
                    p_minus[i] -= h
                    fp = numpy_ln_like_fd(p_plus)
                    fm = numpy_ln_like_fd(p_minus)
                    if fp <= -1e29 and fm <= -1e29:
                        grad[i] = 0.0
                    elif fp <= -1e29:
                        grad[i] = (f0 - fm) / h
                    elif fm <= -1e29:
                        grad[i] = (fp - f0) / h
                    else:
                        grad[i] = (fp - fm) / (2 * h)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e3:
                    grad = grad * (1e3 / grad_norm)
                return grad

            @jax.custom_vjp
            def jax_ln_like(params_arr):
                return jax.pure_callback(
                    lambda p: numpy_ln_like_fd(np.asarray(p)),
                    jnp.float32(0.0),
                    params_arr,
                )

            def jax_ln_like_fwd(params_arr):
                val = jax_ln_like(params_arr)
                return val, params_arr

            def jax_ln_like_bwd(params_arr, g):
                grad = jax.pure_callback(
                    lambda p: numpy_grad_ln_like(np.asarray(p)),
                    jnp.zeros(n_params, dtype=jnp.float32),
                    params_arr,
                )
                return (g * grad,)

            jax_ln_like.defvjp(jax_ln_like_fwd, jax_ln_like_bwd)

        # Numpyro model: priors via distributions, likelihood via factor.
        # Gaussian params use LocScaleReparam (non-centered) so that HMC
        # samples z ~ Normal(0,1) then computes x = loc + scale*z.
        # Without this, params like Mp (~1e27 kg) have gradients ~1e-25
        # which are invisible to HMC with an identity mass matrix.
        _gaussian_reparam = {
            name: LocScaleReparam(centered=0)
            for name in fit_info.fit_param_names
            if isinstance(fit_info.all_params[name], _GaussianParam)
        }

        def numpyro_model():
            params_list = []
            for name in fit_info.fit_param_names:
                param = fit_info.all_params[name]
                if isinstance(param, _UniformParam):
                    val = numpyro.sample(
                        name, dist.Uniform(
                            jnp.float32(param.low_lim),
                            jnp.float32(param.high_lim)))
                elif isinstance(param, _GaussianParam):
                    val = numpyro.sample(
                        name, dist.Normal(
                            jnp.float32(param.best_guess),
                            jnp.float32(param.std)))
                else:
                    val = numpyro.sample(
                        name, dist.Normal(
                            jnp.float32(param.best_guess),
                            jnp.float32(max(1.0, abs(param.best_guess)))))
                params_list.append(val)
            params_arr = jnp.stack(params_list)
            ll = jax_ln_like(params_arr)
            numpyro.factor("log_likelihood", ll)

        numpyro_model = numpyro.handlers.reparam(
            numpyro_model, config=_gaussian_reparam)

        # Initial values for the sampler.
        # Gaussian params are reparameterized: the sampled site is
        # "{name}_decentered" ~ Normal(0,1), so init must be standardized.
        init_values = {}
        for name in fit_info.fit_param_names:
            param = fit_info.all_params[name]
            val = param.best_guess
            if isinstance(param, _UniformParam):
                width = param.high_lim - param.low_lim
                eps = 1e-3 * width
                val = np.clip(val, param.low_lim + eps, param.high_lim - eps)
                init_values[name] = jnp.float32(val)
            elif isinstance(param, _GaussianParam):
                init_values[name + "_decentered"] = jnp.float32(0.0)
            else:
                init_values[name] = jnp.float32(val)

        # Numpy log-likelihood for MAP estimation and post-processing
        def numpy_ln_like(params_np):
            params_np = np.asarray(params_np, dtype=np.float64)
            result = self._ln_like(
                params_np, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities)
            if result == -np.inf or (not np.isscalar(result) and not np.all(np.isfinite(result))):
                return np.float64(-1e30)
            if not np.isscalar(result):
                return np.float64(np.sum(result))
            return np.float64(result)

        for i, name in enumerate(fit_info.fit_param_names):
            param = fit_info.all_params[name]
            val = param.best_guess
            if isinstance(param, _UniformParam):
                width = param.high_lim - param.low_lim
                eps = 1e-3 * width
                val = np.clip(val, param.low_lim + eps, param.high_lim - eps)
                init_values[name] = jnp.float32(val)
            elif isinstance(param, _GaussianParam):
                init_values[name + "_decentered"] = jnp.float32(0.0)
            else:
                init_values[name] = jnp.float32(val)

        for name in init_values:
            print(f"    {name} = {float(init_values[name]):.6g}")

        rng_key = jax.random.PRNGKey(rng_seed)
        print(f"Running NUTS: {num_warmup} warmup + {num_samples} samples...")
        nuts_kwargs = dict(
            target_accept_prob=target_accept_prob,
            dense_mass=True,
            max_tree_depth=8,
            find_heuristic_step_size=True,
            init_strategy=numpyro.infer.init_to_value(values=init_values),
        )
        if inverse_mass_matrix is not None:
            nuts_kwargs["inverse_mass_matrix"] = jnp.array(
                inverse_mass_matrix, dtype=jnp.float32)
            print("  Using provided inverse mass matrix")
        kernel = NUTS(numpyro_model, **nuts_kwargs)
        chain_method = "vectorized" if num_chains > 1 else "sequential"
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=True,
                    chain_method=chain_method)
        mcmc.run(rng_key, extra_fields=("potential_energy",))
        try:
            mcmc.print_summary()
        except Exception:
            print("(print_summary requires more samples for diagnostics)")

        # Extract samples in physical (constrained) space
        samples_dict = mcmc.get_samples(group_by_chain=False)
        flat_samples = np.column_stack([
            np.asarray(samples_dict[name])
            for name in fit_info.fit_param_names
        ])
        samples_dict_chain = mcmc.get_samples(group_by_chain=True)
        chain_samples = np.stack([
            np.asarray(samples_dict_chain[name])
            for name in fit_info.fit_param_names
        ], axis=-1)

        # Extract log-probabilities from NUTS (already computed during sampling).
        # potential_energy = -log_prob, so negate it.
        extra = mcmc.get_extra_fields(group_by_chain=False)
        ln_probs = -np.asarray(extra["potential_energy"])

        # Cast samples to float64 for post-processing (NumPy forward model)
        flat_samples = flat_samples.astype(np.float64)
        chain_samples = chain_samples.astype(np.float64)
        best_params_arr = flat_samples[np.argmax(ln_probs)]

        divisors, new_labels = self._get_divisors_labels(
            np.median(flat_samples, axis=0),
            fit_info.fit_param_names)

        write_param_estimates_file(
            flat_samples / divisors,
            best_params_arr / divisors,
            np.max(ln_probs),
            new_labels)

        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors,
            zero_opacities=zero_opacities, ret_best_fit=True)

        results_dict = {
            "best_fit_params": best_params_arr,
            "chain": chain_samples,
            "flatchain": flat_samples,
            "lnprobability": ln_probs.reshape(num_chains, -1),
            "flatlnprobability": ln_probs,
        }

        retrieval_result = RetrievalResult(
            results_dict, "numpyro", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)

        equal_samples = np.copy(flat_samples)
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
                retrieval_result.random_transit_depths.append(
                    transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(
                    eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(
                    np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            # Get pointwise log-likelihoods (for LOO-CV)
            pl = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, lnlike_per_point=True)
            retrieval_result.pointwise_lnlikes.append(
                pl if not np.isscalar(pl) else np.array([pl]))

        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(
            np.array(retrieval_result.pointwise_lnlikes))

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
                      maxiter=None, maxcall=None, nlive=250, dlogz=0.1,
                      num_final_samples=100, zero_opacities=[],
                      startag='stellar_spectra.pkl',
                      include_opacities=None,
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
        startag : str, optional
            Stellar spectrum file to use
        include_opacities : list, optional
            List of opacities to include
        **dynesty_kwargs : keyword arguments to pass to dynesty's NestedSampler

        Returns
        -------
        result : RetrievalResult object
        '''
        self.params_to_lnlike = {}
        eval_stats = {
            "count": 0,
            "total_seconds": 0.0,
            "last_report_count": 0,
        }

        def _record_eval(elapsed_seconds):
            eval_stats["count"] += 1
            eval_stats["total_seconds"] += elapsed_seconds
            count = eval_stats["count"]
            if count <= 10 or count in (25, 50, 100) or count % 500 == 0:
                avg_ms = 1e3 * eval_stats["total_seconds"] / count
                print(
                    f"[timing] evals={count} total={eval_stats['total_seconds']:.3f}s "
                    f"avg={avg_ms:.3f} ms/eval"
                )

        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            tc_kwargs = dict(include_condensation=include_condensation, method=rad_method)
            if include_opacities is not None:
                tc_kwargs["include_opacities"] = include_opacities
            transit_calc = TransitDepthCalculator(**tc_kwargs)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            ec_kwargs = dict(include_condensation=include_condensation, method=rad_method)
            if include_opacities is not None:
                ec_kwargs["include_opacities"] = include_opacities
            eclipse_calc = EclipseDepthCalculator(**ec_kwargs)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        use_jax = (transit_bins is not None or eclipse_bins is not None)
        jax_setup = None

        if use_jax:
            import jax
            import jax.numpy as jnp
            from ._jax_forward_model import UnsupportedJAXFeatureError
            try:
                jax_setup = self._setup_jax_likelihood(
                    fit_info, "Dynesty",
                    transit_calc, transit_bins, transit_depths, transit_errors,
                    eclipse_calc, eclipse_bins, eclipse_depths, eclipse_errors,
                    zero_opacities=zero_opacities)
            except UnsupportedJAXFeatureError as err:
                print("WARNING: JAX forward model unavailable, falling back "
                      "to the NumPy path: {}".format(err))
                use_jax = False
                jax_setup = None

        if use_jax:
            param_names = jax_setup.param_names
            all_param_defaults = jax_setup.all_param_defaults
            default_arr = jax_setup.default_arr
            jax_data = jax_setup.transit_data
            _jax_forward = jax_setup.transit_forward
            _jax_per_point_lnlike = jax_setup.per_point_lnlike
            _jax_best_fit_binned_depths = jax_setup.best_fit_transit_depths

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        if use_jax:
            def dynesty_ln_like(cube):
                t0 = time.perf_counter()
                params_arr = jnp.array(cube, dtype=jnp.float32)
                per_point = np.array(_jax_per_point_lnlike(params_arr))
                _record_eval(time.perf_counter() - t0)
                if np.any(~np.isfinite(per_point)):
                    self.params_to_lnlike[tuple(cube)] = np.full_like(per_point, -np.inf)
                    return -1e100
                self.params_to_lnlike[tuple(cube)] = per_point
                ln_like = float(per_point.sum())
                if np.random.randint(100) == 0:
                    self.last_params = cube
                    self.last_lnprob = ln_like
                    print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
                return ln_like
        else:
            def dynesty_ln_like(cube):
                t0 = time.perf_counter()
                lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info,
                                                 transit_depths, transit_errors,
                                                 eclipse_depths, eclipse_errors,
                                                 zero_opacities=zero_opacities, lnlike_per_point=True)
                _record_eval(time.perf_counter() - t0)
                if not np.isscalar(lnlike_per_point):
                    ln_like = lnlike_per_point.sum()
                else:
                    assert(lnlike_per_point == -np.inf)
                    ln_like = -np.inf

                self.params_to_lnlike[tuple(cube)] = lnlike_per_point
                if np.random.randint(100) == 0:
                    print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
                return ln_like

        num_dim = fit_info._get_num_fit_params()
        print(f"Starting Dynesty with {num_dim} parameters, nlive={nlive}")
        sampler = NestedSampler(dynesty_ln_like, transform_prior, num_dim, bound='multi', nlive=nlive, **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz = dlogz)

        if eval_stats["count"] > 0:
            avg_ms = 1e3 * eval_stats["total_seconds"] / eval_stats["count"]
            print(
                f"[timing] final evals={eval_stats['count']} "
                f"total={eval_stats['total_seconds']:.3f}s avg={avg_ms:.3f} ms/eval"
            )

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

        # JAX vs Python best-fit comparison
        _jax_best_fit_transit_depths = None
        _python_best_fit_transit_depths = None
        _jax_minus_python_diff = None
        _jax_python_max_abs_diff = None
        _jax_python_rms_diff = None
        _jax_python_danger_flag = 0

        if use_jax and best_fit_transit_depths is not None:
            _jax_best_fit_transit_depths = jax_setup.best_fit_transit_depths(
                best_params_arr)
            _python_best_fit_transit_depths = np.asarray(best_fit_transit_depths, dtype=float)

            if _jax_best_fit_transit_depths is not None:
                _jax_minus_python_diff = _jax_best_fit_transit_depths - _python_best_fit_transit_depths
                _jax_python_max_abs_diff = float(np.max(np.abs(_jax_minus_python_diff)))
                _jax_python_rms_diff = float(np.sqrt(np.mean(_jax_minus_python_diff**2)))
                if _jax_python_max_abs_diff > 1e-6:
                    _jax_python_danger_flag = 1
                    print(
                        f"DANGER: JAX and Python best-fit transit models differ by "
                        f"{_jax_python_max_abs_diff * 1e6:.2f} ppm max (>1 ppm); "
                        f"setting jax_python_danger_flag=1 on the retrieval result."
                    )
            _best_fit_transit_depths_serialized = (
                _jax_best_fit_transit_depths
                if _jax_best_fit_transit_depths is not None
                else best_fit_transit_depths
            )
        else:
            _python_best_fit_transit_depths = (
                np.asarray(best_fit_transit_depths, dtype=float)
                if best_fit_transit_depths is not None else None
            )
            _best_fit_transit_depths_serialized = best_fit_transit_depths

        retrieval_result = RetrievalResult(
            result, "dynesty", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            _best_fit_transit_depths_serialized, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)

        retrieval_result.jax_python_danger_flag = _jax_python_danger_flag
        retrieval_result.best_fit_transit_jax_python_max_abs_diff = _jax_python_max_abs_diff
        retrieval_result.best_fit_transit_jax_python_rms_diff = _jax_python_rms_diff
        retrieval_result.best_fit_transit_jax_minus_python_depths = _jax_minus_python_diff
        retrieval_result.python_best_fit_transit_depths = _python_best_fit_transit_depths
        retrieval_result.jax_best_fit_transit_depths = _jax_best_fit_transit_depths

        # Same JAX-vs-Python cross-check for the emission spectrum
        _jax_best_fit_eclipse_depths = None
        _eclipse_max_abs_diff = None
        if use_jax and best_fit_eclipse_depths is not None:
            _jax_best_fit_eclipse_depths = jax_setup.best_fit_eclipse_depths(
                best_params_arr)
            if _jax_best_fit_eclipse_depths is not None:
                _ediff = (_jax_best_fit_eclipse_depths
                          - np.asarray(best_fit_eclipse_depths, dtype=float))
                _eclipse_max_abs_diff = float(np.max(np.abs(_ediff)))
                retrieval_result.best_fit_eclipse_jax_python_max_abs_diff = _eclipse_max_abs_diff
                retrieval_result.best_fit_eclipse_jax_python_rms_diff = float(
                    np.sqrt(np.mean(_ediff ** 2)))
                retrieval_result.jax_best_fit_eclipse_depths = _jax_best_fit_eclipse_depths
                if _eclipse_max_abs_diff > 1e-6:
                    retrieval_result.jax_python_danger_flag = 1
                    print(
                        f"DANGER: JAX and Python best-fit eclipse models differ by "
                        f"{_eclipse_max_abs_diff * 1e6:.2f} ppm max (>1 ppm); "
                        f"setting jax_python_danger_flag=1 on the retrieval result."
                    )

        retrieval_result.random_transit_depths = []
        retrieval_result.random_eclipse_depths = []
        retrieval_result.random_TP_profiles = []
        retrieval_result.pointwise_lnlikes = []
        for params in equal_samples[:num_final_samples]:
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, ret_best_fit=True)
            # An eclipse-only retrieval never produces transit_info, so only
            # skip a sample when the info it is actually needed for is missing.
            if transit_depths is not None and transit_info is None:
                continue
            if eclipse_depths is not None and eclipse_info is None:
                continue
            if transit_depths is not None:
                retrieval_result.random_transit_depths.append(transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            retrieval_result.pointwise_lnlikes.append(self.params_to_lnlike[tuple(params)])

        #Calculate LOO-CV scores
        try:
            if len(retrieval_result.pointwise_lnlikes) > 1:
                retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
            else:
                retrieval_result.loo_total = retrieval_result.loos = retrieval_result.loo_ks = None
        except Exception as e:
            print(f"LOO-CV skipped: {e}")
            retrieval_result.loo_total = retrieval_result.loos = retrieval_result.loo_ks = None

        return retrieval_result


    def run_multinest(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      maxiter=None, maxcall=None, nlive=250,
                      num_final_samples=100, zero_opacities=[],
                      startag='stellar_spectra.pkl',
                      include_opacities=None,
                      basename=None, resume=False,
                      loglike_callback=None, dump_callback=None,
                      **dynesty_kwargs):
        import pymultinest

        self.params_to_lnlike = {}
        eval_stats = {
            "count": 0,
            "total_seconds": 0.0,
            "last_report_count": 0,
        }

        def _record_eval(elapsed_seconds):
            eval_stats["count"] += 1
            eval_stats["total_seconds"] += elapsed_seconds
            count = eval_stats["count"]
            if count <= 10 or count in (25, 50, 100) or count % 500 == 0:
                avg_ms = 1e3 * eval_stats["total_seconds"] / count
                print(
                    f"[timing] evals={count} total={eval_stats['total_seconds']:.3f}s "
                    f"avg={avg_ms:.3f} ms/eval"
                )

        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            tc_kwargs = dict(include_condensation=include_condensation, method=rad_method)
            if include_opacities is not None:
                tc_kwargs["include_opacities"] = include_opacities
            transit_calc = TransitDepthCalculator(**tc_kwargs)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            ec_kwargs = dict(include_condensation=include_condensation, method=rad_method)
            if include_opacities is not None:
                ec_kwargs["include_opacities"] = include_opacities
            eclipse_calc = EclipseDepthCalculator(**ec_kwargs)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        use_jax = (transit_bins is not None or eclipse_bins is not None)
        jax_setup = None

        if use_jax:
            import jax
            import jax.numpy as jnp
            from ._jax_forward_model import UnsupportedJAXFeatureError
            try:
                jax_setup = self._setup_jax_likelihood(
                    fit_info, "MultiNest",
                    transit_calc, transit_bins, transit_depths, transit_errors,
                    eclipse_calc, eclipse_bins, eclipse_depths, eclipse_errors,
                    zero_opacities=zero_opacities)
            except UnsupportedJAXFeatureError as err:
                print("WARNING: JAX forward model unavailable, falling back "
                      "to the NumPy path: {}".format(err))
                use_jax = False
                jax_setup = None

        if use_jax:
            param_names = jax_setup.param_names
            all_param_defaults = jax_setup.all_param_defaults
            default_arr = jax_setup.default_arr
            jax_data = jax_setup.transit_data
            _jax_forward = jax_setup.transit_forward
            _jax_per_point_lnlike = jax_setup.per_point_lnlike
            _jax_best_fit_binned_depths = jax_setup.best_fit_transit_depths

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        if use_jax:
            def multinest_ln_like(cube):
                t0 = time.perf_counter()
                if loglike_callback is not None:
                    loglike_callback()


                params_arr = jnp.array(cube, dtype=jnp.float32)
                per_point = np.array(_jax_per_point_lnlike(params_arr))
                _record_eval(time.perf_counter() - t0)
                if np.any(~np.isfinite(per_point)):
                    self.params_to_lnlike[tuple(cube)] = np.full_like(per_point, -np.inf)
                    return -1e100
                self.params_to_lnlike[tuple(cube)] = per_point
                ln_like = float(per_point.sum())
                if np.random.randint(100) == 0:
                    self.last_params = cube
                    self.last_lnprob = ln_like
                    print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
                return ln_like
        else:
            def multinest_ln_like(cube):
                t0 = time.perf_counter()
                if loglike_callback is not None:
                    loglike_callback()
                lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                        eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
                _record_eval(time.perf_counter() - t0)
                if not np.isscalar(lnlike_per_point):
                    ln_like = lnlike_per_point.sum()
                    self.params_to_lnlike[tuple(cube)] = lnlike_per_point
                else:
                    # _within_limits returned False → -np.inf sentinel
                    ln_like = -np.inf
                    self.params_to_lnlike[tuple(cube)] = np.array([-np.inf])

                if np.random.randint(100) == 0:
                    print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
                return ln_like

        num_dim = fit_info._get_num_fit_params()
        if basename is None:
            basename = "multinest_" + str(np.random.randint(1000))
        # log_zero: MultiNest treats points with loglike < log_zero as zero-probability.
        # Our bad-point sentinel is -1e100.  The default (-1e90) already handles that
        # correctly (-1e100 < -1e90), so we do NOT override it to -1e100 — that would
        # place our sentinel exactly at the boundary, causing ambiguous treatment.
        solve_kwargs = dict(
            LogLikelihood=multinest_ln_like, Prior=transform_prior,
            n_dims=num_dim, outputfiles_basename=basename,
            verbose=True, resume=resume, n_live_points=nlive)
        if dump_callback is not None:
            solve_kwargs["dump_callback"] = dump_callback
        solve_kwargs.update(dynesty_kwargs)
        result = pymultinest.solve(**solve_kwargs)
        if eval_stats["count"] > 0:
            avg_ms = 1e3 * eval_stats["total_seconds"] / eval_stats["count"]
            print(
                f"[timing] final evals={eval_stats['count']} "
                f"total={eval_stats['total_seconds']:.3f}s avg={avg_ms:.3f} ms/eval"
            )
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

        python_best_fit_transit_depths = None if best_fit_transit_depths is None else np.asarray(best_fit_transit_depths, dtype=float)
        jax_best_fit_transit_depths = None
        best_fit_transit_source = "python_full_output"
        jax_python_danger_flag = 0
        best_fit_transit_jax_python_max_abs_diff = None
        best_fit_transit_jax_python_rms_diff = None
        best_fit_transit_jax_minus_python_depths = None
        if use_jax and transit_depths is not None:
            jax_best_fit_transit_depths = _jax_best_fit_binned_depths(best_params_arr)
            if jax_best_fit_transit_depths is not None:
                # Always use JAX binned depths as the serialized best fit
                best_fit_transit_depths = jax_best_fit_transit_depths
                if python_best_fit_transit_depths is None:
                    best_fit_transit_source = "jax_binned_only"
                else:
                    _diff = jax_best_fit_transit_depths - python_best_fit_transit_depths
                    _max_abs_diff = float(np.max(np.abs(_diff)))
                    best_fit_transit_jax_python_max_abs_diff = _max_abs_diff
                    best_fit_transit_jax_python_rms_diff = float(np.sqrt(np.mean(_diff ** 2)))
                    best_fit_transit_jax_minus_python_depths = _diff
                    if _max_abs_diff > 1e-6:
                        jax_python_danger_flag = 1
                        print(
                            f"DANGER: JAX and Python best-fit transit models differ by "
                            f"{1e6 * _max_abs_diff:.3f} ppm max (>1 ppm); setting "
                            "jax_python_danger_flag=1 on the retrieval result."
                        )
                        best_fit_transit_source = "jax_binned_python_info_preserved"
                    else:
                        best_fit_transit_source = "jax_python_consistent"
                    # Always preserve best_fit_transit_info when Python full-output exists

        # Fallback to median params if best-fit fell outside limits
        if best_fit_transit_depths is None and transit_depths is not None:
            print("WARNING: best-fit params outside limits, falling back to median of posterior")
            median_params = np.median(equal_samples, axis=0)
            best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
                median_params,
                transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)
            best_params_arr = median_params

        retrieval_result = RetrievalResult(
            result, "pymultinest", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)
        retrieval_result.best_fit_transit_source = best_fit_transit_source
        retrieval_result.jax_python_danger_flag = jax_python_danger_flag
        if best_fit_transit_jax_python_max_abs_diff is not None:
            retrieval_result.best_fit_transit_jax_python_max_abs_diff = best_fit_transit_jax_python_max_abs_diff
            retrieval_result.best_fit_transit_jax_python_rms_diff = best_fit_transit_jax_python_rms_diff
            retrieval_result.best_fit_transit_jax_minus_python_depths = best_fit_transit_jax_minus_python_depths
        if python_best_fit_transit_depths is not None:
            retrieval_result.python_best_fit_transit_depths = python_best_fit_transit_depths
        if jax_best_fit_transit_depths is not None:
            retrieval_result.jax_best_fit_transit_depths = jax_best_fit_transit_depths

        retrieval_result.random_transit_depths = []
        retrieval_result.random_eclipse_depths = []
        retrieval_result.random_TP_profiles = []
        retrieval_result.pointwise_lnlikes = []
        for params in equal_samples[:num_final_samples]:
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors, ret_best_fit=True)
            # An eclipse-only retrieval never produces transit_info, so only
            # skip a sample when the info it is actually needed for is missing.
            if transit_depths is not None and transit_info is None:
                continue
            if eclipse_depths is not None and eclipse_info is None:
                continue
            if transit_depths is not None:
                retrieval_result.random_transit_depths.append(transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            retrieval_result.pointwise_lnlikes.append(self.params_to_lnlike[tuple(params)])

        #Calculate LOO-CV scores
        try:
            if len(retrieval_result.pointwise_lnlikes) > 1:
                retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
            else:
                retrieval_result.loo_total = retrieval_result.loos = retrieval_result.loo_ks = None
        except Exception as e:
            print(f"LOO-CV skipped: {e}")
            retrieval_result.loo_total = retrieval_result.loos = retrieval_result.loo_ks = None
        return retrieval_result


    def run_ultranest(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      nlive=400, nsteps=None,
                      num_final_samples=100, zero_opacities=[],
                      startag='stellar_spectra.pkl',
                      include_opacities=None,
                      log_dir=None, resume='resume',
                      **ultranest_kwargs):
        """Run UltraNest nested sampling retrieval.

        UltraNest uses MLFriends algorithm which handles multimodal and curved
        degeneracies better than MultiNest's ellipsoidal decomposition.

        Parameters
        ----------
        nlive : int
            Minimum number of live points (default 400)
        nsteps : int, optional
            Number of slice sampler steps. If provided, uses SliceSampler which
            handles curved degeneracies better. Recommended: 2 * n_params
        log_dir : str, optional
            Directory for UltraNest output files
        resume : str
            Resume mode: 'overwrite', 'resume', 'resume-similar', 'subfolder'
        **ultranest_kwargs
            Additional arguments passed to ReactiveNestedSampler
        """
        import ultranest
        import ultranest.stepsampler

        self.params_to_lnlike = {}
        eval_stats = {
            "count": 0,
            "total_seconds": 0.0,
            "last_report_count": 0,
        }

        def _record_eval(elapsed_seconds):
            eval_stats["count"] += 1
            eval_stats["total_seconds"] += elapsed_seconds
            count = eval_stats["count"]
            if count <= 10 or count in (25, 50, 100) or count % 500 == 0:
                avg_ms = 1e3 * eval_stats["total_seconds"] / count
                print(
                    f"[timing] evals={count} total={eval_stats['total_seconds']:.3f}s "
                    f"avg={avg_ms:.3f} ms/eval"
                )

        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            tc_kwargs = dict(include_condensation=include_condensation, method=rad_method)
            if include_opacities is not None:
                tc_kwargs["include_opacities"] = include_opacities
            transit_calc = TransitDepthCalculator(**tc_kwargs)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            ec_kwargs = dict(include_condensation=include_condensation, method=rad_method)
            if include_opacities is not None:
                ec_kwargs["include_opacities"] = include_opacities
            eclipse_calc = EclipseDepthCalculator(**ec_kwargs)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        use_jax = (transit_bins is not None or eclipse_bins is not None)
        jax_setup = None

        if use_jax:
            import jax
            import jax.numpy as jnp
            from ._jax_forward_model import UnsupportedJAXFeatureError
            try:
                jax_setup = self._setup_jax_likelihood(
                    fit_info, "UltraNest",
                    transit_calc, transit_bins, transit_depths, transit_errors,
                    eclipse_calc, eclipse_bins, eclipse_depths, eclipse_errors,
                    zero_opacities=zero_opacities)
            except UnsupportedJAXFeatureError as err:
                print("WARNING: JAX forward model unavailable, falling back "
                      "to the NumPy path: {}".format(err))
                use_jax = False
                jax_setup = None

        if use_jax:
            param_names = jax_setup.param_names
            all_param_defaults = jax_setup.all_param_defaults
            default_arr = jax_setup.default_arr
            jax_data = jax_setup.transit_data
            _jax_forward = jax_setup.transit_forward
            _jax_per_point_lnlike = jax_setup.per_point_lnlike
            _jax_best_fit_binned_depths = jax_setup.best_fit_transit_depths

        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = fit_info._from_unit_interval(i, cube[i])
            return new_cube

        if use_jax:
            def ultranest_ln_like(cube):
                t0 = time.perf_counter()
                params_arr = jnp.array(cube, dtype=jnp.float32)
                per_point = np.array(_jax_per_point_lnlike(params_arr))
                _record_eval(time.perf_counter() - t0)
                if np.any(~np.isfinite(per_point)):
                    self.params_to_lnlike[tuple(cube)] = np.full_like(per_point, -np.inf)
                    return -1e100
                self.params_to_lnlike[tuple(cube)] = per_point
                ln_like = float(per_point.sum())
                if np.random.randint(100) == 0:
                    self.last_params = cube
                    self.last_lnprob = ln_like
                    print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
                return ln_like
        else:
            def ultranest_ln_like(cube):
                t0 = time.perf_counter()
                lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info,
                                                 transit_depths, transit_errors,
                                                 eclipse_depths, eclipse_errors,
                                                 zero_opacities=zero_opacities, lnlike_per_point=True)
                _record_eval(time.perf_counter() - t0)
                if not np.isscalar(lnlike_per_point):
                    ln_like = lnlike_per_point.sum()
                else:
                    assert(lnlike_per_point == -np.inf)
                    ln_like = -np.inf

                self.params_to_lnlike[tuple(cube)] = lnlike_per_point
                if np.random.randint(100) == 0:
                    print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
                return ln_like

        num_dim = fit_info._get_num_fit_params()
        if log_dir is None:
            log_dir = "ultranest_" + str(np.random.randint(10000))

        print(f"Starting UltraNest with {num_dim} parameters, nlive={nlive}")
        if nsteps is not None:
            print(f"Using SliceSampler with nsteps={nsteps} (good for curved degeneracies)")

        sampler = ultranest.ReactiveNestedSampler(
            fit_info.fit_param_names,
            ultranest_ln_like,
            transform_prior,
            log_dir=log_dir,
            resume=resume,
            **ultranest_kwargs
        )

        if nsteps is not None:
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=nsteps,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                region_filter=True
            )

        result = sampler.run(min_num_live_points=nlive)

        if eval_stats["count"] > 0:
            avg_ms = 1e3 * eval_stats["total_seconds"] / eval_stats["count"]
            print(
                f"[timing] final evals={eval_stats['count']} "
                f"total={eval_stats['total_seconds']:.3f}s avg={avg_ms:.3f} ms/eval"
            )

        samples = result['samples']
        logl = result['weighted_samples']['logl']

        best_idx = np.argmax(logl)
        best_params_arr = result['weighted_samples']['points'][best_idx]

        equal_samples = samples.copy()
        np.random.shuffle(equal_samples)

        divisors, new_labels = self._get_divisors_labels(
            np.median(equal_samples, axis=0),
            fit_info.fit_param_names)

        # UltraNest returns logz as scalar, but write_param_estimates_file expects it
        logz_val = result['logz']
        logz_err = result['logzerr']

        write_param_estimates_file(
            equal_samples / divisors,
            best_params_arr / divisors,
            logz_val,
            new_labels)

        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)

        # Convert UltraNest result format to match what RetrievalResult expects
        # RetrievalResult expects logz to be an array (like MultiNest), so wrap scalar
        result_for_retrieval = dict(result)
        result_for_retrieval['logz'] = np.array([logz_val])
        result_for_retrieval['logzerr'] = np.array([logz_err])

        retrieval_result = RetrievalResult(
            result_for_retrieval, "ultranest", best_params_arr,
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
            # An eclipse-only retrieval never produces transit_info, so only
            # skip a sample when the info it is actually needed for is missing.
            if transit_depths is not None and transit_info is None:
                continue
            if eclipse_depths is not None and eclipse_info is None:
                continue
            if transit_depths is not None:
                retrieval_result.random_transit_depths.append(
                    transit_info["unbinned_depths"] * transit_info["unbinned_correction_factors"])
            if eclipse_depths is not None:
                retrieval_result.random_eclipse_depths.append(eclipse_info["unbinned_eclipse_depths"])
                retrieval_result.random_TP_profiles.append(
                    np.array([eclipse_info["P_profile"], eclipse_info["T_profile"]]))
            retrieval_result.pointwise_lnlikes.append(self.params_to_lnlike[tuple(params)])

        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(
            np.array(retrieval_result.pointwise_lnlikes))

        return retrieval_result


    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T=None, logZ=0, CO_ratio=0.53, log_CH4_mult=0,
                             free_retrieval=False,
                             add_H_minus_absorption=False,
                             log_cloudtop_P=np.inf, log_scatt_factor=0,
                             scatt_slope=4, error_multiple=1, error_additive=0, T_star=None,
                             T_spot=None, spot_cov_frac=None,
                             cloud_cov_frac=1.0,
                             frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size=-6,
                             n=None, log_k=-np.inf,
                             log_P_quench=-99, T_quench=None, quench_species=None,
                             scattering_ref_wavelength=1e-6,
                             transit_offset_windows=None,
                             offset_niriss=0, offset_nrs1=0, offset_miri=0,
                             fit_vmr=False, fit_clr=False,
                             log_SO2=None, log_CH4=None, log_TiO=None, log_VO=None, log_CS2=None,
                             log_S=None,
                             limb_asym=False, delta_T=0.0, f_limb=0.5,
                             log_cloudtop_P_evening=np.inf,
                             log_scatt_factor_evening=0,
                             scatt_slope_evening=4,
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
        limb_asym : bool
            If True, use a 1.5D limb-asymmetric model instead of the standard
            or patchy-cloud model.  The morning limb is cloudy (log_cloudtop_P)
            and uses the base T/P profile; the evening limb is cloud-free and
            offset by delta_T.  The two sectors are linearly combined by f_limb.
        delta_T : float
            Temperature offset (K) added to the base T/P profile for the
            evening (clear) limb.  Expected to be positive for hot Jupiters.
        f_limb : float
            Fraction of the transit annulus contributed by the morning (cloudy)
            sector.  Range [0, 1]; default 0.5.
        log_cloudtop_P_evening : float
            log10 cloud-top pressure (Pa) for the evening limb.  Defaults to
            np.inf (no clouds).  The retrieval can push this to high pressures
            (effectively cloud-free) or fit a separate cloud deck.
        log_scatt_factor_evening : float
            log10 of the Rayleigh-like scattering enhancement factor for the
            evening limb. Defaults to 0, which reproduces the previous
            Rayleigh-only treatment.
        scatt_slope_evening : float
            Rayleigh-like scattering slope for the evening limb. Defaults to 4.
        log_S : float or None
            Optional tied free retrieval parameter for sulfur. When not
            None, 10**log_S is the ELEMENTAL sulfur abundance (S atoms
            per total atmospheric molecules), with the S-atom budget
            split 1/3 into H2S and 2/3 into CS2 — which produces equal
            molecular VMRs because CS2 carries 2 S atoms per molecule:
                VMR(H2S) = (1/3) * 10**log_S
                VMR(CS2) = (1/3) * 10**log_S
            log_S is therefore directly comparable to S/H (modulo H VMR).
            Leave as None to disable; the override branch is then
            traced out at JIT compile time (zero runtime cost). Active
            in two paths, both opt-in:
              (a) Equilibrium-chemistry mode (fit_vmr=False, fit_clr=False):
                  overrides the equilibrium H2S and CS2 abundances,
                  matching the log_SO2/log_CH4/log_TiO/log_VO override pattern.
              (b) Free-VMR mode (fit_vmr=True): list "S" in fit_info.gases
                  (alongside "H2-He" and your other free gases) and add
                  log_S as a fit parameter. The bg-VMR calculation is
                  adjusted to subtract only the (2/3) actually consumed
                  by molecules. Other free-VMR gases (CH4, VO, ...) are
                  fit independently as usual.

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
