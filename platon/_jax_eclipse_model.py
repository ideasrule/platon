"""
JAX-native forward model for emission (secondary-eclipse) spectra.

This is the emission counterpart of the transit forward model in
``_jax_forward_model.py``, and reuses all of its machinery: chemistry on the
master (T, P) grid, the hydrostatic solve, the fast temperature slab, the gas
/ CIA / Rayleigh opacity bases, and the stellar spectrum handling.  Only the
radiative-transfer step and the binning differ.

The physics matches ``eclipse_depth_calculator.EclipseDepthCalculator``
exactly:

    tau(lambda, i)  = cumsum_i  0.5*(k_i + k_{i+1}) * dr_i
    F(lambda)       = -2 pi sum_i B(lambda, T_i) * [E3(tau_{i+1}) - E3(tau_i)]
    F              += pi * B(lambda, T_cloud) * g(tau_max)   [if clouds]
    depth(lambda)   = F / F_star * (R_photosphere / R_s)^2

with ``g(t) = t^2 E1(t) - t e^-t + e^-t``.  Clouds are handled by masking
layers below the cloud top (fixed array shapes, JIT-friendly) rather than by
truncating the pressure profile as the NumPy code does; the two are
equivalent.

Patchy clouds: when ``cloud_cov_frac`` is being fit (or set to anything
other than 1), the depth is the area-weighted sum of a cloudy and a clear
column, ``f * depth_cloudy + (1 - f) * depth_clear``, each with its own
photosphere radius.  The two columns share the chemistry, T/P profile and
opacities; only the radiative transfer is done twice.

Not supported (fall back to the NumPy calculator): correlated-k, Mie
scattering, and the rocky-surface (``surface_type`` / ``surface_pressure``)
branch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from .constants import h, c, k_B
from ._jax_forward_model import (
    _FM_DTYPE, _k_B, _h, _c, _PI, _EPS,
    _jax_build_atmosphere_state,
    _jax_get_temperature_slab,
    _jax_compute_opacity_bases_auto,
    _jax_add_scattering,
    _jax_interpolate_to_profile,
    _jax_interpolate_to_profile_isothermal,
    _jax_compute_stellar_spectra,
    prepare_jax_data,
    UnsupportedJAXFeatureError,
    _fit_info_value,
)

# E_3(tau) lookup table.  Identical nodes and out-of-range values to
# EclipseDepthCalculator.tau_cache / exp3_cache, so the JAX and NumPy models
# make the same discretisation error.
_TAU_CACHE = np.logspace(-6, 3, 1000)
_EXP3_CACHE = scipy.special.expn(3, _TAU_CACHE)

# t^2 * E_1(t), tabulated directly because E_1 diverges at t -> 0 while the
# product does not.  Used only for the cloud-deck emission term.
_X2E1_X = np.logspace(-10, 4, 5000)
_X2E1_Y = _X2E1_X ** 2 * scipy.special.expn(1, _X2E1_X)

_HC = float(h * c)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _raise_if_unsupported_eclipse_options(atm_solver, fit_info):
    unsupported = []
    if getattr(atm_solver, "method", "xsec") != "xsec":
        unsupported.append("method != 'xsec' (ktables path)")
    if _fit_info_value(fit_info, "surface_type", None) is not None:
        unsupported.append("surface_type (rocky-surface emission)")
    surface_P = _fit_info_value(fit_info, "surface_pressure", np.inf)
    if surface_P is not None and np.isfinite(float(surface_P)):
        unsupported.append("finite surface_pressure")
    if unsupported:
        raise UnsupportedJAXFeatureError(
            "Unsupported options for the JAX emission model: "
            + ", ".join(unsupported))


def prepare_jax_eclipse_data(atm_solver, abundance_getter, wavelength_bins,
                             T_star, T_spot, spot_cov_frac, stellar_blackbody,
                             fit_info, n_data_points, zero_opacities=None):
    """Pack static arrays for the emission model.

    Builds on :func:`_jax_forward_model.prepare_jax_data` and replaces the
    binning weights: eclipse depths are averaged with the stellar *photon*
    spectrum as weights (``F_star * lambda / hc``), matching
    ``EclipseDepthCalculator._get_binned_depths``.
    """
    _raise_if_unsupported_eclipse_options(atm_solver, fit_info)

    data = prepare_jax_data(
        atm_solver, abundance_getter, wavelength_bins,
        T_star, T_spot, spot_cov_frac, stellar_blackbody,
        fit_info, n_data_points, zero_opacities=zero_opacities)

    _f = np.float32
    lambda_grid = np.asarray(data["lambda_grid"], dtype=np.float64)
    stellar_spectrum = np.asarray(data["stellar_spectrum"], dtype=np.float64)

    # Photon spectrum = energy spectrum / (hc/lambda)
    photon_factor = lambda_grid / _HC
    photon_spectrum = stellar_spectrum * photon_factor

    expanded_idx = np.asarray(data["expanded_lambda_idx"])
    bin_ids = np.asarray(data["bin_ids"])
    n_bins = int(data["n_bins"])

    w = photon_spectrum[expanded_idx]
    sums = np.zeros(n_bins, dtype=np.float64)
    np.add.at(sums, bin_ids, w)
    w = w / np.maximum(sums[bin_ids], 1e-300)

    data["eclipse_bin_weights_1d"] = jnp.array(w, dtype=_f)
    data["photon_factor"] = jnp.array(photon_factor, dtype=_f)
    data["tau_cache"] = jnp.array(_TAU_CACHE, dtype=_f)
    data["exp3_cache"] = jnp.array(_EXP3_CACHE, dtype=_f)
    data["x2e1_x"] = jnp.array(_X2E1_X, dtype=_f)
    data["x2e1_y"] = jnp.array(_X2E1_Y, dtype=_f)

    # ---- eclipse offset window -------------------------------------------
    # EclipseDepthCalculator users apply a single constant offset over
    # [offset_start:offset_end].  Bake it into a mask when present.
    offset_start = _fit_info_value(fit_info, "offset_start", None)
    offset_end = _fit_info_value(fit_info, "offset_end", None)
    has_offset = (
        "offset_eclipse" in getattr(fit_info, "all_params", {})
        and offset_start is not None and offset_end is not None)
    if has_offset:
        mask = np.zeros(n_bins, dtype=np.float64)
        lo = max(0, min(int(offset_start), n_bins))
        hi = max(0, min(int(offset_end), n_bins))
        mask[lo:hi] = 1.0
    else:
        mask = np.zeros(n_bins, dtype=np.float64)
    data["eclipse_offset_mask"] = jnp.array(mask, dtype=_f)
    data["has_eclipse_offset"] = bool(has_offset)

    # ---- patchy clouds -----------------------------------------------------
    # Decided from fit_param_names as well as the best guess: the transit
    # model's "patchy" flag looks only at the best guess, so fitting
    # cloud_cov_frac with a guess of exactly 1.0 would silently disable it.
    cloud_cov_guess = _fit_info_value(fit_info, "cloud_cov_frac", 1.0)
    data["eclipse_patchy"] = bool(
        "cloud_cov_frac" in getattr(fit_info, "fit_param_names", [])
        or (cloud_cov_guess is not None and float(cloud_cov_guess) != 1.0))

    return data


# ---------------------------------------------------------------------------
# Radiative transfer primitives
# ---------------------------------------------------------------------------

def _jax_exp3(tau, tau_cache, exp3_cache):
    """E_3(tau), interpolated exactly as EclipseDepthCalculator._exp3 does."""
    return jnp.interp(tau, tau_cache, exp3_cache,
                      left=_FM_DTYPE(0.5), right=_FM_DTYPE(0.0))


def _jax_cloud_emission_factor(tau, x2e1_x, x2e1_y):
    """g(tau) = tau^2 E_1(tau) - tau e^-tau + e^-tau.

    Fraction of an opaque deck's emission that escapes through an overlying
    slab of optical depth ``tau``.  g(0) = 1, g(inf) = 0.
    """
    x2e1 = jnp.interp(tau, x2e1_x, x2e1_y,
                      left=_FM_DTYPE(0.0), right=_FM_DTYPE(0.0))
    exp_neg = jnp.exp(-tau)
    return x2e1 - tau * exp_neg + exp_neg


def _jax_planck(lambda_grid, temperatures):
    """B_lambda(T), shape (N_lambda, N_T_points).

    Written as exp(-x) / -expm1(-x) instead of 1 / (exp(x) - 1) so that the
    Wien tail underflows to zero in float32 instead of overflowing to inf.
    """
    lam = lambda_grid[:, None]
    x = _h * _c / (lam * _k_B * temperatures[None, :])
    neg_exp = jnp.exp(-x)
    return (2 * _h * _c ** 2 / lam ** 5) * neg_exp / jnp.maximum(
        -jnp.expm1(-x), _EPS)


def _jax_eclipse_bin(depths, params_dict, data, stellar_spectrum):
    """Photon-weighted binning plus the optional constant eclipse offset."""
    with jax.named_scope("eclipse_binning"):
        expanded = depths[data["expanded_lambda_idx"]]
        if data["dynamic_stellar"]:
            weights = (stellar_spectrum * data["photon_factor"])[
                data["expanded_lambda_idx"]]
            weight_sums = jax.ops.segment_sum(
                weights, data["bin_ids"], num_segments=data["n_bins"])
            weights = weights / jnp.maximum(
                weight_sums[data["bin_ids"]], _EPS)
        else:
            weights = data["eclipse_bin_weights_1d"]
        binned = jax.ops.segment_sum(
            expanded * weights, data["bin_ids"], num_segments=data["n_bins"])

        if data.get("has_eclipse_offset", False):
            offset = _FM_DTYPE(params_dict.get("offset_eclipse", 0.0))
            binned = binned + offset * data["eclipse_offset_mask"]
        return binned


# ---------------------------------------------------------------------------
# Full forward model:  params -> binned eclipse depths
# ---------------------------------------------------------------------------

def _jax_eclipse_core(params_dict, data, state=None):
    """Shared body: params -> (unbinned depths, stellar spectrum, diagnostics).

    ``state`` is the tuple returned by ``_jax_build_atmosphere_state``; pass it
    in to avoid recomputing the chemistry and hydrostatic solve.
    """
    scatt_factor = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_scatt_factor"])
    scatt_slope = _FM_DTYPE(params_dict["scatt_slope"])
    log_cloudtop_P = _FM_DTYPE(params_dict["log_cloudtop_P"])

    if state is None:
        state = _jax_build_atmosphere_state(params_dict, data)
    (Rs, T_grid, P_grid, P_profile, T_profile, abundances,
     radii, dr, is_unbound) = state
    Rp = _FM_DTYPE(params_dict["Rp"])

    stellar_spectrum, _ = _jax_compute_stellar_spectra(params_dict, data)

    is_isothermal = data.get("profile_type", "isothermal") == "isothermal"
    use_fast_t, t_start = _jax_get_temperature_slab(
        T_profile, T_grid, data.get("disable_fast_t", False))

    with jax.named_scope("temperature_slab_selection"):
        base_absorption_coeff, scattering_base = _jax_compute_opacity_bases_auto(
            abundances, data, T_grid, P_grid, use_fast_t, t_start)

    absorption_coeff = _jax_add_scattering(
        base_absorption_coeff, scattering_base,
        data["lambda_grid"], scatt_factor, scatt_slope,
        data["ln_scatt_lambda_ratio"])

    with jax.named_scope("interpolate_to_layers"):
        interp = (_jax_interpolate_to_profile_isothermal if is_isothermal
                  else _jax_interpolate_to_profile)
        absorption_coeff_atm = interp(
            absorption_coeff, T_grid, P_grid, T_profile, P_profile,
            data["profile_p_lo"], data["profile_p_hi"],
            data["profile_p_frac"], data["ln_P_profile"],
            data["ln_P_grid"], data["ln_kBT_grid"])

    fluxes, photosphere_radii, integrand, taus, coeff_masked = _jax_eclipse_flux(
        absorption_coeff_atm, T_profile, radii, dr, Rp, log_cloudtop_P, data)

    unbinned = (fluxes / jnp.maximum(stellar_spectrum, _EPS)
                * (photosphere_radii / Rs) ** 2)

    patchy_info = {}
    if data.get("eclipse_patchy", False):
        # Clear column: same opacities, no cloud deck.
        cloud_cov_frac = _FM_DTYPE(params_dict.get("cloud_cov_frac", 1.0))
        fluxes_clear, photosphere_radii_clear, _, _, _ = _jax_eclipse_flux(
            absorption_coeff_atm, T_profile, radii, dr, Rp,
            _FM_DTYPE(jnp.inf), data)
        unbinned_clear = (fluxes_clear / jnp.maximum(stellar_spectrum, _EPS)
                          * (photosphere_radii_clear / Rs) ** 2)
        patchy_info = {
            "cloud_cov_frac": cloud_cov_frac,
            "unbinned_eclipse_depths_cloudy": unbinned,
            "unbinned_eclipse_depths_clear": unbinned_clear,
        }
        unbinned = (cloud_cov_frac * unbinned
                    + (1 - cloud_cov_frac) * unbinned_clear)

    info = {
        **patchy_info,
        "unbinned_eclipse_depths": unbinned,
        "unbinned_wavelengths": data["lambda_grid"],
        "planet_spectrum": fluxes,
        "stellar_spectrum": stellar_spectrum,
        "absorption_coeff_atm": coeff_masked,
        "P_profile": data["P_profile"],
        "T_profile": T_profile,
        "radii": radii,
        "dr": dr,
        "taus": taus,
        "photosphere_radii": photosphere_radii,
        "contrib": -integrand / fluxes[:, None],
    }
    return unbinned, stellar_spectrum, info


def jax_compute_eclipse_depths(params_dict, data, return_full=False):
    """Compute binned eclipse depths.  Pure JAX, JIT-compatible.

    Set ``return_full=True`` to also get the unbinned depths, planet flux,
    optical depths, T/P profile and contribution function.
    """
    with jax.named_scope("eclipse_forward_model"):
        if return_full:
            unbinned, stellar_spectrum, info = _jax_eclipse_core(
                params_dict, data)
            binned = _jax_eclipse_bin(unbinned, params_dict, data,
                                      stellar_spectrum)
            return binned, info

        # Sampling path: short-circuit unbound atmospheres to NaN so the
        # sampler rejects them without running the radiative transfer.
        state = _jax_build_atmosphere_state(params_dict, data)
        is_unbound = state[-1]

        def unbound_branch(_):
            return jnp.full((data["n_bins"],), jnp.nan, dtype=_FM_DTYPE)

        def bound_branch(_):
            unbinned, stellar_spectrum, _info = _jax_eclipse_core(
                params_dict, data, state=state)
            return _jax_eclipse_bin(unbinned, params_dict, data,
                                    stellar_spectrum)

        return jax.lax.cond(is_unbound, unbound_branch, bound_branch,
                            operand=None)


def _jax_eclipse_flux(absorption_coeff_atm, T_profile, radii, dr, Rp,
                      log_cloudtop_P, data):
    """Planet emergent flux, photosphere radii, and the flux integrand."""
    lambda_grid = data["lambda_grid"]

    # ---- cloud deck: mask layers at or below the cloud top ----------------
    # The NumPy path truncates P_profile at the cloud top; masking keeps array
    # shapes static for JIT and gives the same integral.
    log10_P = data["log10_P_profile"]
    active = log10_P < log_cloudtop_P
    n_active = jnp.clip(jnp.sum(active).astype(jnp.int32), 2, radii.shape[0])
    absorption_coeff_atm = jnp.where(
        active[:, None], absorption_coeff_atm, _FM_DTYPE(0.0))
    # Shell i spans layers i and i+1; keep only shells fully above the cloud.
    shell_mask = jnp.arange(dr.shape[0]) < (n_active - 1)

    intermediate_coeff = 0.5 * (absorption_coeff_atm[:-1]
                                + absorption_coeff_atm[1:])
    intermediate_T = 0.5 * (T_profile[:-1] + T_profile[1:])
    d_taus = jnp.where(shell_mask[None, :],
                       intermediate_coeff.T * dr[None, :], _FM_DTYPE(0.0))
    taus = jnp.cumsum(d_taus, axis=1)

    planck = _jax_planck(lambda_grid, intermediate_T)
    padded_taus = jnp.concatenate(
        [jnp.zeros((taus.shape[0], 1), dtype=_FM_DTYPE), taus], axis=1)
    e3 = _jax_exp3(padded_taus, data["tau_cache"], data["exp3_cache"])
    integrand = planck * jnp.diff(e3, axis=1)
    fluxes = -2 * _PI * jnp.sum(integrand, axis=1)

    max_taus = taus[:, -1]

    # Opaque cloud deck radiating through the overlying column.
    cloud_idx = jnp.clip(n_active - 2, 0, intermediate_T.shape[0] - 1)
    planck_cloud = jnp.take(planck, cloud_idx, axis=1)
    cloud_flux = _PI * planck_cloud * _jax_cloud_emission_factor(
        max_taus, data["x2e1_x"], data["x2e1_y"])
    fluxes = fluxes + jnp.where(
        jnp.isfinite(log_cloudtop_P), cloud_flux, _FM_DTYPE(0.0))

    # Radius where tau is closest to 1; Rp if the column is optically thin.
    log_taus = jnp.log(jnp.maximum(taus, _EPS))
    idx = jnp.argmin(jnp.abs(log_taus), axis=1)
    photosphere_radii = jnp.where(max_taus < _FM_DTYPE(1.0), Rp, radii[idx])

    return (fluxes, photosphere_radii, integrand, taus, absorption_coeff_atm)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def jax_eclipse_log_likelihood(params_arr, param_names, all_param_defaults,
                               data, measured_depths, measured_errors):
    """Scalar emission log-likelihood from a flat parameter vector."""
    params_dict = dict(all_param_defaults)
    for i, name in enumerate(param_names):
        params_dict[name] = params_arr[i]

    binned = jax_compute_eclipse_depths(params_dict, data)
    err_mult = _FM_DTYPE(params_dict["error_multiple"])
    scaled_err = err_mult * measured_errors.astype(_FM_DTYPE)
    resid = binned - measured_depths.astype(_FM_DTYPE)
    per_point = resid ** 2 / scaled_err ** 2 + jnp.log(
        _FM_DTYPE(2 * jnp.pi) * scaled_err ** 2)
    return -0.5 * jnp.sum(per_point)
