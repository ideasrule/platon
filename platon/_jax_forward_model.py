"""
JAX-native forward model for differentiable transit depth computation.

Replaces the NumPy forward model + finite-difference gradients with a pure JAX
implementation that supports automatic differentiation via jax.grad.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as scipy_special

from .constants import k_B, AMU, G, M_sun, Teff_sun, h, c
from .params import NUM_LAYERS

# Forward model precision.  Float32 is ~6x faster on GPU (RTX 4090).
_FM_DTYPE = jnp.float32

# Physical constants cast to forward-model dtype (avoid repeated casts)
_k_B = _FM_DTYPE(k_B)
_AMU = _FM_DTYPE(AMU)
_G   = _FM_DTYPE(G)
_h   = _FM_DTYPE(h)
_c   = _FM_DTYPE(c)
_PI  = _FM_DTYPE(np.pi)

# Minimum clamp value for float32.  Must be above float32 min normal (1.18e-38)
# so that log(_EPS) is finite and 1/_EPS doesn't overflow.
# 1e-30 → log = -69, 1/eps = 1e30, both within float32 range (±3.4e38).
_EPS = _FM_DTYPE(1e-30)
_LN_EPS = _FM_DTYPE(np.log(1e-30))
_LN_10 = _FM_DTYPE(np.log(10.0))
_LN_K_B = _FM_DTYPE(np.log(k_B))
_LN_MIN_CROSS_SEC = _FM_DTYPE(np.log(1e-99))
_LOG10_MIN_ABUNDANCE = _FM_DTYPE(-99.0)
_RAYLEIGH_CONST = _FM_DTYPE(128.0 / 3 * np.pi ** 5)
_SCATT_REF = _FM_DTYPE(1e-6)
_LN_SCATT_REF = _FM_DTYPE(np.log(1e-6))
_POL_SQR_SCALE = _FM_DTYPE(1e30)
_POL_SQR_INV_SCALE = _FM_DTYPE(1e-30)

# Fast path for narrow parametric T/P profiles. 
_FAST_T_GRID_SPAN = 8

# H- bound-free / free-free cross-section constants (John 1988)
_H_MINUS_ALPHA = np.float64(14391.0)
_H_MINUS_LAMBDA0 = np.float64(1.6419)  # microns
_H_MINUS_BF_C = np.array(
    [152.519, 49.534, -118.858, 92.536, -34.194, 4.982], dtype=np.float64)
_H_MINUS_FF_RED = np.array([
    [0, 0, 0, 0, 0, 0],
    [2483.346, 285.827, -2054.291, 2827.776, -1341.537, 208.952],
    [-3449.889, -1158.382, 8746.523, -11485.632, 5303.609, -812.939],
    [2200.04, 2427.719, -13651.105, 16755.524, -7510.494, 1132.738],
    [-696.271, -1841.4, 8624.97, -10051.53, 4400.067, -655.02],
    [88.283, 444.517, -1863.864, 2095.288, -901.788, 132.985]],
    dtype=np.float64)
_H_MINUS_FF_MID = np.array([
    [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
    [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.36],
    [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
    [115.5291, 169.6374, -245.649, 114.243, -21.9972, 1.5097],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]],
    dtype=np.float64)


def _compute_H_minus_cross_section(T_grid, lambda_grid):
    """Precompute H- bound-free + free-free cross-section k(T, lambda).

    Vectorised NumPy version of _atmosphere_solver._get_k, evaluated for
    every temperature in T_grid at once.  Returns (N_T, N_lambda) in SI
    units (m^4/N) ready to be multiplied by n_el * n_H * P^2 / (k_B * T).
    """
    wl = lambda_grid * 1e6  # metres -> microns
    alpha = _H_MINUS_ALPHA
    lam0 = _H_MINUS_LAMBDA0
    N_T = len(T_grid)
    N_lam = len(wl)

    # --- bound-free: only for wavelengths < lambda_0 -----------------------
    bf_mask = wl < lam0
    k_bf = np.zeros((N_T, N_lam), dtype=np.float64)
    if np.any(bf_mask):
        wl_bf = wl[bf_mask]
        inv_wl = 1.0 / wl_bf
        inv_lam0 = 1.0 / lam0
        diff = inv_wl - inv_lam0
        powers = diff[np.newaxis, :] ** (np.arange(6)[:, np.newaxis] / 2.0)
        f_lambda = np.dot(_H_MINUS_BF_C, powers)  # (N_bf,)
        sigma = 1e-18 * wl_bf ** 3 * diff ** 1.5 * f_lambda  # (N_bf,)
        for t in range(N_T):
            T = T_grid[t]
            k_bf[t, bf_mask] = (0.75 * T ** (-2.5)
                                * np.exp(alpha / lam0 / T)
                                * (1 - np.exp(-alpha / wl_bf / T))
                                * sigma)

    # --- free-free: two wavelength regimes ----------------------------------
    k_ff = np.zeros((N_T, N_lam), dtype=np.float64)
    mid_mask = (wl > 0.1823) & (wl < 0.3645)
    red_mask = wl > 0.3645

    for n in range(1, 7):
        theta_pow = (5040.0 / T_grid) ** ((n + 1) / 2.0)  # (N_T,)
        if np.any(mid_mask):
            wl_mid = wl[mid_mask]
            A_mid = np.column_stack([wl_mid ** p for p in (2, 0, -1, -2, -3, -4)])
            k_ff[:, mid_mask] += 1e-29 * theta_pow[:, np.newaxis] * A_mid.dot(_H_MINUS_FF_MID[n - 1])
        if np.any(red_mask):
            wl_red = wl[red_mask]
            A_red = np.column_stack([wl_red ** p for p in (2, 0, -1, -2, -3, -4)])
            k_ff[:, red_mask] += 1e-29 * theta_pow[:, np.newaxis] * A_red.dot(_H_MINUS_FF_RED[n - 1])

    # cm^4/dyne -> m^4/N
    return (k_bf + k_ff) * 1e-3


# E_2(x) lookup table for the Line et al. (2013) radiative-solution T/P
# profile.  Linearly interpolated in x (like the E_3 cache in the emission
# model) over a log-spaced grid, with E_2(0) = 1 and E_2(inf) = 0 as the
# out-of-range values.
_EXPN2_X = np.logspace(-10, 3, 4000)
_EXPN2_Y = scipy_special.expn(2, _EXPN2_X)


def _to_numpy(arr, dtype=np.float64):
    """Convert CuPy or NumPy array to NumPy array."""
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.array(arr, dtype=dtype)


# ---------------------------------------------------------------------------
# Interpolation primitives (JAX ports of _interpolator_3D.py)
# ---------------------------------------------------------------------------

def jax_interp1d(target_xs, xs, data):
    """JAX port of _interpolator_3D.interp1d.

    Linear interpolation along axis 0 of ``data``.
    """
    target_xs = jnp.atleast_1d(jnp.asarray(target_xs, dtype=_FM_DTYPE))
    x_indices = jnp.interp(target_xs, xs, jnp.arange(len(xs), dtype=_FM_DTYPE))
    x_lo = jnp.floor(x_indices).astype(int)
    x_hi = jnp.minimum(x_lo + 1, len(xs) - 1)
    frac = x_indices - x_lo

    if data.ndim > 1:
        frac = frac.reshape((-1,) + (1,) * (data.ndim - 1))

    return data[x_lo] * (1 - frac) + data[x_hi] * frac


def jax_regular_grid_interp(ys, xs, data, target_ys, target_xs):
    """JAX port of _interpolator_3D.regular_grid_interp.

    Bilinear interpolation on a regular grid.
    ys: (Ny,), xs: (Nx,) -- sorted grid vectors
    data: (Ny, Nx, ...)
    target_ys, target_xs: (M,) or scalar  (must have same length)
    """
    target_ys = jnp.atleast_1d(jnp.asarray(target_ys, dtype=_FM_DTYPE))
    target_xs = jnp.atleast_1d(jnp.asarray(target_xs, dtype=_FM_DTYPE))

    x_indices = jnp.interp(target_xs, xs, jnp.arange(len(xs), dtype=_FM_DTYPE))
    y_indices = jnp.interp(target_ys, ys, jnp.arange(len(ys), dtype=_FM_DTYPE))

    x_lo = jnp.floor(x_indices).astype(int)
    x_hi = jnp.minimum(x_lo + 1, len(xs) - 1)
    x_frac = x_indices - x_lo

    y_lo = jnp.floor(y_indices).astype(int)
    y_hi = jnp.minimum(y_lo + 1, len(ys) - 1)
    y_frac = y_indices - y_lo

    # Reshape fractions for broadcasting with extra data dimensions
    if data.ndim > 2 and target_ys.shape[0] > 0:
        extra = data.ndim - 2
        x_frac = x_frac.reshape((-1,) + (1,) * extra)
        y_frac = y_frac.reshape((-1,) + (1,) * extra)

    return (data[y_lo, x_lo] * (1 - y_frac) * (1 - x_frac) +
            data[y_hi, x_lo] * y_frac * (1 - x_frac) +
            data[y_lo, x_hi] * (1 - y_frac) * x_frac +
            data[y_hi, x_hi] * y_frac * x_frac)


def _jax_blackbody_stellar_spectrum(temperature, lambda_grid):
    """Match AtmosphereSolver.get_stellar_spectrum() blackbody branch."""
    exponent = _h * _c / (lambda_grid * _k_B * temperature)
    return (2 * _c * _PI / lambda_grid ** 4
            / jnp.expm1(exponent)
            * _h * _c / lambda_grid)


# ---------------------------------------------------------------------------
# Hydrostatic solver (JAX port of _hydrostatic_solver.py)
# ---------------------------------------------------------------------------

def _jax_get_radii(ln_Ps, planet_mass, planet_radius,
                   ln_P_profile, T_profile, mu_profile):
    """Compute radii from hydrostatic equilibrium."""
    intermediate_mu = (
        jnp.interp(ln_Ps[1:], ln_P_profile, mu_profile) +
        jnp.interp(ln_Ps[:-1], ln_P_profile, mu_profile)) / 2
    intermediate_T = (
        jnp.interp(ln_Ps[1:], ln_P_profile, T_profile) +
        jnp.interp(ln_Ps[:-1], ln_P_profile, T_profile)) / 2
    d_inv_r = (jnp.diff(ln_Ps) * _k_B * intermediate_T /
               (_G * planet_mass * intermediate_mu * _AMU))
    inv_r = 1.0 / planet_radius + jnp.cumsum(d_inv_r)
    return jnp.concatenate([jnp.array([planet_radius]), 1.0 / inv_r])


def _jax_hydrostatic_solve(ln_P_profile, T_profile, mu_profile,
                           planet_mass, planet_radius,
                           ln_P_below, ln_P_above):
    """JAX port of _hydrostatic_solver._solve.

    Uses precomputed ln_P_below / ln_P_above arrays (fixed shapes) to avoid
    boolean indexing inside JIT.
    """
    radii_below = _jax_get_radii(ln_P_below, planet_mass, planet_radius,
                                 ln_P_profile, T_profile, mu_profile)
    radii_above = _jax_get_radii(ln_P_above, planet_mass, planet_radius,
                                 ln_P_profile, T_profile, mu_profile)

    radii = jnp.concatenate([radii_above[1:][::-1], radii_below[1:]])
    dr = -jnp.diff(radii)
    return radii, dr


def _jax_atmosphere_is_unbound(T_profile, mu_profile, planet_mass,
                               planet_radius, star_radius, T_star=None):
    """Match the legacy Hill-radius rejection in _hydrostatic_solver._solve."""
    if T_star is None:
        T_star = _FM_DTYPE(Teff_sun)

    R_hill = (star_radius
              * (T_star / T_profile[0]) ** 2
              * (planet_mass / (_FM_DTYPE(3.0) * _FM_DTYPE(M_sun))) ** _FM_DTYPE(1.0 / 3.0))
    denom = (1.0 / planet_radius
             + _k_B * jnp.median(T_profile) * jnp.log(_FM_DTYPE(1e-4))
             / (_G * planet_mass * jnp.mean(mu_profile) * _AMU))
    max_r_estimate = 1.0 / denom
    return jnp.logical_or(max_r_estimate < 0, max_r_estimate > R_hill)


# ---------------------------------------------------------------------------
# Optical depth (JAX port of _tau_calculator.py)
# ---------------------------------------------------------------------------

def _jax_get_dl(radii):
    """Compute path-length matrix.  radii must be in descending order."""
    radii_sq = radii * radii
    sqr_length = radii_sq[:, None] - radii_sq[None, 1:]
    sqr_length = jnp.maximum(sqr_length, _FM_DTYPE(0.0))
    lengths = 2 * jnp.sqrt(sqr_length)
    return lengths[:-1] - lengths[1:]


def _jax_get_line_of_sight_tau_from_dl(absorption_coeff, dl):
    """Line-of-sight optical depth.

    absorption_coeff : (N_layers, N_lambda)
    dl               : (N_layers-1, N_layers-1)
    Returns          : (N_lambda, N_layers)
    """
    intermediate_coeff = 0.5 * (absorption_coeff[:-1] + absorption_coeff[1:])
    return jnp.dot(intermediate_coeff.T, dl)


def _jax_get_line_of_sight_tau(absorption_coeff, radii):
    """Line-of-sight optical depth."""
    return _jax_get_line_of_sight_tau_from_dl(absorption_coeff, _jax_get_dl(radii))


# ---------------------------------------------------------------------------
# Data preparation  (NumPy -- called once before sampling)
# ---------------------------------------------------------------------------

class UnsupportedJAXFeatureError(RuntimeError):
    """Raised when a feature enabled in fit_info has no JAX implementation."""
    pass


def _get_fit_info_bool(fit_info, param_name, invert=False):
    """Extract a boolean flag from fit_info, handling param wrappers."""
    if not hasattr(fit_info, 'all_params'):
        return False
    p = fit_info.all_params.get(param_name)
    if p is None:
        return False
    val = p.best_guess if hasattr(p, 'best_guess') else p
    if invert:
        # For cloud_cov_frac: patchy mode if not exactly 1.0
        return float(val) != 1.0
    return bool(val)
def _fit_info_value(fit_info, name, default=None):
    if not hasattr(fit_info, "all_params"):
        return default
    p = fit_info.all_params.get(name)
    if p is None:
        return default
    return p.best_guess if hasattr(p, "best_guess") else p


def _raise_if_unsupported_jax_options(atm_solver, fit_info):
    unsupported = []

    # Existing explicit guards
    profile_type = _fit_info_value(fit_info, "profile_type", "isothermal")
    if profile_type == "twopoint":
        unsupported.append(f"profile_type='{profile_type}'")
    if profile_type == "radiative_solution":
        for required in ("a", "beta", "log_k_th", "log_gamma"):
            if _fit_info_value(fit_info, required, None) is None:
                unsupported.append(
                    f"profile_type='radiative_solution' without '{required}'")

    mie_n = _fit_info_value(fit_info, "n", None)
    if mie_n is not None:
        unsupported.append("Mie scattering (n/ri != None)")

    # Legacy options that are not clearly threaded through JAX
    if getattr(atm_solver, "method", "xsec") != "xsec":
        unsupported.append("method != 'xsec' (ktables path)")

    # custom_abundances / custom_T_profile / custom_P_profile are supported
    # via photochem mode — skip the guard when photochem_mode is set.
    if not getattr(fit_info, "photochem_mode", False):
        if _fit_info_value(fit_info, "custom_abundances", None) is not None:
            unsupported.append("custom_abundances")
        if _fit_info_value(fit_info, "custom_T_profile", None) is not None:
            unsupported.append("custom_T_profile")
        if _fit_info_value(fit_info, "custom_P_profile", None) is not None:
            unsupported.append("custom_P_profile")

    if _fit_info_value(fit_info, "add_gas_absorption", True) is False:
        unsupported.append("add_gas_absorption=False")

    if _fit_info_value(fit_info, "add_collisional_absorption", True) is False:
        unsupported.append("add_collisional_absorption=False")

    if _fit_info_value(fit_info, "add_scattering", True) is False:
        unsupported.append("add_scattering=False")

    scatt_ref = _fit_info_value(fit_info, "scattering_ref_wavelength", 1e-6)
    if scatt_ref is not None and float(scatt_ref) != 1e-6:
        unsupported.append("scattering_ref_wavelength != 1e-6")

    min_abund = _fit_info_value(fit_info, "min_abundance", 1e-99)
    if min_abund is not None and float(min_abund) != 1e-99:
        unsupported.append(f"min_abundance={min_abund}")

    min_xsec = _fit_info_value(fit_info, "min_cross_sec", 1e-99)
    if min_xsec is not None and float(min_xsec) != 1e-99:
        unsupported.append(f"min_cross_sec={min_xsec}")

    if unsupported:
        raise UnsupportedJAXFeatureError(
            "Unsupported options for JAX forward model: " + ", ".join(unsupported)
        )

def validate_runtime_params_python(params_dict, data):
    # temperature profile
    T_profile = np.asarray(_jax_get_temperature_profile(params_dict, data), dtype=float)
    if T_profile.min() < data["min_temperature"] or T_profile.max() > data["max_temperature"]:
        raise ValueError(
            f"Invalid temperatures in T/P profile: "
            f"{T_profile.min():.1f} to {T_profile.max():.1f} K"
        )

    # only relevant for equilibrium-table chemistry
    if not data["vmr_mode"] and not data["clr_mode"]:
        logZ = float(params_dict["logZ"])
        if logZ < data["logZ_min"] or logZ > data["logZ_max"]:
            raise ValueError(f"logZ {logZ} is out of bounds")

        co = float(params_dict["CO_ratio"])
        if co < data["CO_min"] or co > data["CO_max"]:
            raise ValueError(f"CO_ratio {co} is out of bounds")

    # cloudtop
    log_ctp = float(params_dict["log_cloudtop_P"])
    if np.isfinite(log_ctp):
        P = 10.0 ** log_ctp
        if P <= data["P_min"] or P > data["P_max"]:
            raise ValueError(f"Cloudtop pressure {P} Pa is out of bounds")

    abundances = np.asarray(_jax_get_master_grid_abundances(params_dict, data, T_profile))
    _, mu_profile = _jax_interp_abundances_to_profile(
        jnp.asarray(abundances),
        data["all_masses"],
        data["T_grid"],
        data["P_grid"],
        jnp.asarray(T_profile, dtype=_FM_DTYPE),
        data["P_profile"],
        data["profile_p_lo"],
        data["profile_p_hi"],
        data["profile_p_frac"],
    )
    T_star = params_dict.get("T_star")
    is_unbound = bool(_jax_atmosphere_is_unbound(
        jnp.asarray(T_profile, dtype=_FM_DTYPE),
        mu_profile,
        _FM_DTYPE(params_dict["Mp"]),
        _FM_DTYPE(params_dict["Rp"]),
        _FM_DTYPE(params_dict["Rs"]),
        None if T_star is None else _FM_DTYPE(T_star),
    ))
    if is_unbound:
        raise ValueError("Atmosphere unbound: height > hill radius")


def validate_runtime_params_jax(params_dict, data):
    """Cheap JAX-native validity mask for the hot likelihood path.

    Keep only checks that are scientifically important and not naturally
    rejected by the JAX forward model itself. Unbound atmospheres are handled
    inside the forward model and produce NaNs there, so that case is
    intentionally not duplicated here.
    """
    T_profile = _jax_get_temperature_profile(params_dict, data)
    Tmin = jnp.min(T_profile)
    Tmax = jnp.max(T_profile)
    if data.get("limb_asym", False):
        # Evening profile is a rigid delta_T shift of the base profile. Make
        # sure both sides stay within the opacity-grid temperature range.
        delta_T_lim = _FM_DTYPE(params_dict.get("delta_T", 0.0))
        Tmin = jnp.minimum(Tmin, Tmin + delta_T_lim)
        Tmax = jnp.maximum(Tmax, Tmax + delta_T_lim)
    valid_temperature = (
        jnp.isfinite(Tmin)
        & jnp.isfinite(Tmax)
        & (Tmin >= _FM_DTYPE(data["min_temperature"]))
        & (Tmax <= _FM_DTYPE(data["max_temperature"]))
    )

    if (not data["vmr_mode"] and not data["clr_mode"]
            and not data.get("photochem_mode", False)):
        logZ = _FM_DTYPE(params_dict["logZ"])
        co = _FM_DTYPE(params_dict["CO_ratio"])
        valid_chemistry = (
            jnp.isfinite(logZ)
            & jnp.isfinite(co)
            & (logZ >= _FM_DTYPE(data["logZ_min"]))
            & (logZ <= _FM_DTYPE(data["logZ_max"]))
            & (co >= _FM_DTYPE(data["CO_min"]))
            & (co <= _FM_DTYPE(data["CO_max"]))
        )
    else:
        valid_chemistry = jnp.bool_(True)

    log_ctp = _FM_DTYPE(params_dict["log_cloudtop_P"])
    cloud_pressure = _FM_DTYPE(10.0) ** log_ctp
    valid_cloudtop = jnp.logical_or(
        ~jnp.isfinite(log_ctp),
        (cloud_pressure > _FM_DTYPE(data["P_min"]))
        & (cloud_pressure <= _FM_DTYPE(data["P_max"]))
    )
    if data.get("limb_asym", False):
        log_ctp_eve = _FM_DTYPE(params_dict.get("log_cloudtop_P_evening", jnp.inf))
        cloud_pressure_eve = _FM_DTYPE(10.0) ** log_ctp_eve
        valid_cloudtop_eve = jnp.logical_or(
            ~jnp.isfinite(log_ctp_eve),
            (cloud_pressure_eve > _FM_DTYPE(data["P_min"]))
            & (cloud_pressure_eve <= _FM_DTYPE(data["P_max"]))
        )
        valid_cloudtop = valid_cloudtop & valid_cloudtop_eve

    Rs = _FM_DTYPE(params_dict["Rs"])
    Mp = _FM_DTYPE(params_dict["Mp"])
    Rp = _FM_DTYPE(params_dict["Rp"])
    err_mult = _FM_DTYPE(params_dict["error_multiple"])
    valid_scalars = (
        jnp.isfinite(Rs) & (Rs > 0)
        & jnp.isfinite(Mp) & (Mp > 0)
        & jnp.isfinite(Rp) & (Rp > 0)
        & jnp.isfinite(err_mult) & (err_mult > 0)
    )
    if "error_additive" in params_dict:
        err_add = _FM_DTYPE(params_dict["error_additive"])
        valid_scalars = valid_scalars & jnp.isfinite(err_add) & (err_add >= 0)

    return valid_temperature & valid_chemistry & valid_cloudtop & valid_scalars
        

def prepare_jax_data(atm_solver, abundance_getter, wavelength_bins,
                     T_star, T_spot, spot_cov_frac, stellar_blackbody,
                     fit_info, n_data_points, zero_opacities=None):
    """Pack all static arrays into a dict of JAX arrays."""
    _raise_if_unsupported_jax_options(atm_solver, fit_info)
    '''
    # ---- guard: unsupported features that would be silently dropped --------
    if hasattr(fit_info, 'all_params'):
        _pt = fit_info.all_params.get("profile_type")
        if _pt is not None:
            _ptv = _pt.best_guess if hasattr(_pt, 'best_guess') else str(_pt)
            if _ptv in ("radiative_solution", "twopoint"):
                raise UnsupportedJAXFeatureError(
                    f"profile_type='{_ptv}' is not implemented in the JAX "
                    f"forward model. Use the non-JAX path or switch to "
                    f"'isothermal' / 'parametric'.")
        _ri = fit_info.all_params.get("n")
        if _ri is not None:
            _riv = _ri.best_guess if hasattr(_ri, 'best_guess') else _ri
            if _riv is not None:
                raise UnsupportedJAXFeatureError(
                    "Mie scattering (n != None) is not implemented in the "
                    "JAX forward model. Use the non-JAX path or disable Mie "
                    "scattering (n=None).")
    '''
    T_grid = _to_numpy(atm_solver.T_grid)
    P_grid = _to_numpy(atm_solver.P_grid)
    lambda_grid_full = _to_numpy(atm_solver.lambda_grid)
    P_profile = np.logspace(np.log10(P_grid[0]), np.log10(P_grid[-1]),
                            NUM_LAYERS)
    ln_P_grid = np.log(P_grid)
    ln_P_profile = np.log(P_profile)
    ln_kBT_grid = np.log(k_B * T_grid)
    n_dens_grid = np.exp(ln_P_grid[None, :] - ln_kBT_grid[:, None])
    p_profile_indices = np.interp(
        ln_P_profile, ln_P_grid, np.arange(len(P_grid), dtype=np.float64))
    profile_p_lo = np.floor(p_profile_indices).astype(np.int32)
    profile_p_hi = np.minimum(profile_p_lo + 1, len(P_grid) - 1).astype(np.int32)
    profile_p_frac = (p_profile_indices - profile_p_lo).astype(np.float32)

    # ---- Trim wavelength grid to observation range -----------------------
    # Only keep wavelengths that fall within the range covered by bins,
    # with a small buffer.  This avoids computing opacities at wavelengths
    # that will never contribute to any bin.
    if wavelength_bins is not None:
        wb = np.array(wavelength_bins)
        wl_min = wb.min()
        wl_max = wb.max()
        # Add 1% buffer on each side for safety
        buf = 0.01 * (wl_max - wl_min)
        trim_lo = max(0, np.searchsorted(lambda_grid_full, wl_min - buf))
        trim_hi = min(len(lambda_grid_full),
                      np.searchsorted(lambda_grid_full, wl_max + buf))
        lambda_grid = lambda_grid_full[trim_lo:trim_hi]
        print(f"  Trimmed wavelength grid: {len(lambda_grid_full)} → {len(lambda_grid)} "
              f"({100*len(lambda_grid)/len(lambda_grid_full):.1f}%)")
    else:
        trim_lo = 0
        trim_hi = len(lambda_grid_full)
        lambda_grid = lambda_grid_full

    # ---- abundance tables ------------------------------------------------
    logZs = _to_numpy(abundance_getter.logZs)
    CO_ratios = _to_numpy(abundance_getter.CO_ratios)
    log_abundances = _to_numpy(abundance_getter.log_abundances)
    log_abundances = np.where(np.isfinite(log_abundances), log_abundances, -99.0)
    species_names = list(abundance_getter.included_species)
    species_names += ["DMS", "CS2", "CH3SH"]
    print(species_names)
    all_masses = np.array(
        [float(_to_numpy(np.atleast_1d(atm_solver.mass_data.get(s, 0.0)))[0])
         for s in species_names],
       dtype=np.float64)

    # ---- gas opacity data (stacked, trimmed in wavelength) ---------------
    zero_opacities = set(zero_opacities or [])
    opacity_species_indices = []
    absorption_arrays = []
    for s in species_names:
        if s in zero_opacities:
            continue
        if s in atm_solver.absorption_data:
            opacity_species_indices.append(species_names.index(s))
            arr = _to_numpy(atm_solver.absorption_data[s])
            absorption_arrays.append(arr[:, :, trim_lo:trim_hi])

    if absorption_arrays:
        all_absorption_data = np.stack(absorption_arrays, axis=0)
    else:
        all_absorption_data = np.zeros(
            (0, len(T_grid), len(P_grid), len(lambda_grid)), dtype=np.float64)
    opacity_species_indices = np.array(opacity_species_indices, dtype=np.int32)

    # ---- Rayleigh polarizabilities ---------------------------------------
    all_pol_sqr = np.zeros(len(species_names), dtype=np.float64)
    for i, s in enumerate(species_names):
        if s in atm_solver.polarizability_data:
            all_pol_sqr[i] = (float(atm_solver.polarizability_data[s]) ** 2
                              * float(_POL_SQR_SCALE))

    # ---- collisional absorption (trimmed in wavelength) ------------------
    coll_data_list, coll_idx_list = [], []
    for (s1, s2) in atm_solver.collisional_absorption_data:
        if s1 in species_names and s2 in species_names:
            coll_idx_list.append([species_names.index(s1),
                                  species_names.index(s2)])
            arr = _to_numpy(atm_solver.collisional_absorption_data[(s1, s2)])
            coll_data_list.append(arr[:, trim_lo:trim_hi])
    if coll_data_list:
        all_coll_data = np.stack(coll_data_list, axis=0) / (1e-15 * 1e-15)
        coll_indices = np.array(coll_idx_list, dtype=np.int32)
    else:
        all_coll_data = np.zeros((0, len(T_grid), len(lambda_grid)),
                                 dtype=np.float64)
        coll_indices = np.zeros((0, 2), dtype=np.int32)

    # ---- H- bound-free / free-free absorption -----------------------------
    add_H_minus = False
    if hasattr(fit_info, 'all_params'):
        hm = fit_info.all_params.get("add_H_minus_absorption")
        if hm is not None:
            add_H_minus = bool(hm.best_guess if hasattr(hm, 'best_guess') else hm)
    el_index = species_names.index("el") if "el" in species_names else -1
    H_index = species_names.index("H") if "H" in species_names else -1
    if add_H_minus and (el_index < 0 or H_index < 0):
        print("WARNING: add_H_minus_absorption=True but 'el' or 'H' not in "
              "species list — disabling H-")
        add_H_minus = False
    if add_H_minus:
        # Precompute k(T, lambda) on the full T_grid x lambda_grid (trimmed).
        # Shape: (N_T, N_lambda).  This is constant across the entire retrieval.
        h_minus_cross = _compute_H_minus_cross_section(T_grid, lambda_grid)
        print(f"  H- absorption enabled: cross-section grid "
              f"({h_minus_cross.shape[0]}x{h_minus_cross.shape[1]})")
    else:
        h_minus_cross = np.zeros((0, 0), dtype=np.float64)

    # ---- stellar spectrum & binning weights (trimmed) --------------------
    stellar_spectrum_full, correction_factors_full = atm_solver.get_stellar_spectrum(
        T_star, T_spot, spot_cov_frac, stellar_blackbody)
    stellar_spectrum_full = _to_numpy(stellar_spectrum_full)
    correction_factors_full = _to_numpy(correction_factors_full)
    stellar_spectrum = stellar_spectrum_full[trim_lo:trim_hi]
    correction_factors = correction_factors_full[trim_lo:trim_hi]

    n_bins = len(wavelength_bins) if wavelength_bins is not None else len(lambda_grid)
    # Sparse binning via segment_sum.  Indices are now relative to the
    # trimmed lambda_grid.
    if wavelength_bins is not None:
        expanded_lambda_idx = []
        expanded_bin_ids = []
        expanded_weights = []
        for i, (start, end) in enumerate(np.array(wavelength_bins)):
            mask = np.logical_and(lambda_grid > start, lambda_grid < end)
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                raise ValueError(f"Wavelength bin too narrow: {start}-{end} meters")
            if len(idx) <= 5:
                print(f"WARNING: only {len(idx)} points in {start}-{end} m bin. Results will be inaccurate")

            w = stellar_spectrum[idx].copy()
            wsum = w.sum()
            if wsum > 0:
                w /= wsum
            expanded_lambda_idx.append(idx.astype(np.int32))
            expanded_bin_ids.append(np.full(len(idx), i, dtype=np.int32))
            expanded_weights.append(w)
        expanded_lambda_idx = np.concatenate(expanded_lambda_idx)
        bin_ids = np.concatenate(expanded_bin_ids)
        bin_weights_1d = np.concatenate(expanded_weights)
    else:
        expanded_lambda_idx = np.arange(len(lambda_grid))
        bin_ids = np.arange(len(lambda_grid), dtype=np.int32)
        bin_weights_1d = np.ones(len(lambda_grid), dtype=np.float64)

    # ---- hydrostatic split (precompute fixed-shape arrays) ---------------
    ref_pressure = float(atm_solver.ref_pressure)
    P_below_arr = np.concatenate([[ref_pressure],
                                  P_profile[P_profile > ref_pressure]])
    P_above_arr = np.concatenate([[ref_pressure],
                                  P_profile[P_profile <= ref_pressure][::-1]])

    # ---- offset mask -----------------------------------------------------
    param_names = list(fit_info.fit_param_names)
    defaults = fit_info._interpret_param_array(
        np.array([fit_info.all_params[n].best_guess for n in param_names]))

    # ---- dynamic offset masks from transit_offset_windows ----------------
    offset_names = []
    offset_masks_list = []
    transit_offset_windows = None
    if hasattr(fit_info, 'all_params'):
        tow = fit_info.all_params.get("transit_offset_windows")
        if tow is not None:
            transit_offset_windows = tow.best_guess if hasattr(tow, 'best_guess') else tow
    if isinstance(transit_offset_windows, dict) and transit_offset_windows:
        for offset_name, (start, end) in transit_offset_windows.items():
            if offset_name in fit_info.fit_param_names or offset_name in fit_info.all_params:
                mask = np.zeros(n_bins, dtype=np.float64)
                start = max(0, min(int(start), n_bins))
                end = max(0, min(int(end), n_bins))
                mask[start:end] = 1.0
                offset_names.append(offset_name)
                offset_masks_list.append(mask)
    if offset_masks_list:
        offset_masks = np.stack(offset_masks_list, axis=0)  # (n_offsets, n_bins)
    else:
        offset_masks = np.zeros((0, n_bins), dtype=np.float64)

    # ---- species indices for individual abundance overrides ---------------
    ch4_index = species_names.index("CH4") if "CH4" in species_names else -1
    so2_index = species_names.index("SO2") if "SO2" in species_names else -1
    tio_index = species_names.index("TiO") if "TiO" in species_names else -1
    vo_index = species_names.index("VO") if "VO" in species_names else -1

    # ---- VMR / CLR mode setup --------------------------------------------
    vmr_mode = bool(fit_info.all_params["fit_vmr"].best_guess)
    clr_mode = bool(fit_info.all_params["fit_clr"].best_guess)
    vmr_gas_names = []   # Python list of strings
    vmr_mapping = []     # list of (species_idx, gas_idx, frac) – all Python values

    if vmr_mode or clr_mode:
        vmr_gas_names = list(fit_info.gases)
        for i, g in enumerate(vmr_gas_names):
            if g in species_names:
                vmr_mapping.append((species_names.index(g), i, 1.0))
            elif g == "H2-He":
                # Split into H2 and He using the effective mass to infer ratio
                h2_idx = species_names.index("H2")
                he_idx = species_names.index("He")
                mass_H2 = float(atm_solver.mass_data["H2"])
                mass_He = float(atm_solver.mass_data["He"])
                mass_H2He = float(atm_solver.mass_data["H2-He"])
                h2_frac = (mass_He - mass_H2He) / (mass_He - mass_H2)
                he_frac = 1.0 - h2_frac
                vmr_mapping.append((h2_idx, i, h2_frac))
                vmr_mapping.append((he_idx, i, he_frac))
            elif g == "S":
                # Tied sulfur free retrieval (toggleable).
                # 10**log_S is interpreted as the ELEMENTAL S abundance
                # (S atoms per total atmospheric molecules). S atoms are
                # split 1/3 into H2S and 2/3 into CS2, which gives equal
                # molecular VMRs because CS2 has 2 S atoms per molecule:
                #     VMR(H2S) = (1/3) * 10**log_S
                #     VMR(CS2) = (1/3) * 10**log_S
                # Total molecular VMR consumed = (2/3) * 10**log_S; this
                # is accounted for in the bg calc via vmr_gas_total_frac.
                if "H2S" not in species_names or "CS2" not in species_names:
                    raise ValueError(
                        "Tied-sulfur gas 'S' requires both H2S and CS2 to be "
                        "in the species list (load their opacities).")
                h2s_idx = species_names.index("H2S")
                cs2_idx = species_names.index("CS2")
                vmr_mapping.append((h2s_idx, i, 1.0 / 3.0))
                vmr_mapping.append((cs2_idx, i, 1.0 / 3.0))
            else:
                raise ValueError(
                    f"VMR gas '{g}' not found in species list and is not "
                    f"'H2-He' or 'S'")
    if vmr_mapping:
        vmr_species_indices = np.array([m[0] for m in vmr_mapping], dtype=np.int32)
        vmr_gas_indices = np.array([m[1] for m in vmr_mapping], dtype=np.int32)
        vmr_fracs = np.array([m[2] for m in vmr_mapping], dtype=np.float32)
    else:
        vmr_species_indices = np.zeros(0, dtype=np.int32)
        vmr_gas_indices = np.zeros(0, dtype=np.int32)
        vmr_fracs = np.zeros(0, dtype=np.float32)

    # Per-gas sum of mapping fractions, used to compute the actual
    # molecular VMR consumed by each fit gas in the bg calculation.
    # For normal gases (one species, frac=1.0) this is 1.0; for 'H2-He'
    # it is h2_frac + he_frac = 1.0; for the tied 'S' gas it is 2/3
    # (since 10**log_S is the elemental S abundance, not the molecular
    # VMR sum). All-1.0 default makes this a no-op for any pre-existing
    # gas list.
    if vmr_mode or clr_mode:
        vmr_gas_total_frac = np.ones(len(vmr_gas_names), dtype=np.float32)
        frac_sum_by_gas = {}
        for sp_idx, g_idx, f in vmr_mapping:
            frac_sum_by_gas[g_idx] = frac_sum_by_gas.get(g_idx, 0.0) + f
        for g_idx, total in frac_sum_by_gas.items():
            vmr_gas_total_frac[g_idx] = total
    else:
        vmr_gas_total_frac = np.zeros(0, dtype=np.float32)

    # ---- Photochem mode --------------------------------------------------
    # Activated when PhotochemRetriever attaches arrays to fit_info
    photochem_mode = getattr(fit_info, "photochem_mode", False)
    if photochem_mode:
        pc_base_abund = getattr(fit_info, "photochem_base_abundances")  # {name: (N_T,N_P)}
        pc_T_raw      = np.asarray(getattr(fit_info, "photochem_pc_T"), dtype=np.float64)
        pc_P_raw      = np.asarray(getattr(fit_info, "photochem_pc_P"), dtype=np.float64)
        pc_mols       = list(getattr(fit_info, "photochem_retrievable_mols"))

        # Interpolate photochem T onto PLATON's internal P_profile (log-P space)
        pc_T_on_profile = np.interp(
            np.log(P_profile), np.log(pc_P_raw), pc_T_raw,
            left=float(pc_T_raw[0]), right=float(pc_T_raw[-1])
        ).astype(np.float32)

        # Stack base abundances into (n_species, N_T, N_P) float32
        pc_abund_stack = np.zeros(
            (len(species_names), len(T_grid), len(P_grid)), dtype=np.float32)
        for mol, arr in pc_base_abund.items():
            if mol in species_names:
                pc_abund_stack[species_names.index(mol)] = arr.astype(np.float32)

        # Per-molecule species indices (-1 if not in species list)
        pc_mol_indices = np.array(
            [species_names.index(m) if m in species_names else -1
             for m in pc_mols], dtype=np.int32)

        pc_h2_idx = species_names.index("H2") if "H2" in species_names else -1
        pc_he_idx = species_names.index("He") if "He" in species_names else -1
    else:
        pc_T_on_profile = np.zeros(len(P_profile), dtype=np.float32)
        pc_abund_stack  = np.zeros(
            (len(species_names), len(T_grid), len(P_grid)), dtype=np.float32)
        pc_mol_indices  = np.zeros(0, dtype=np.int32)
        pc_mols         = []
        pc_h2_idx       = -1
        pc_he_idx       = -1

    # ---- profile type ------------------------------------------------------
    profile_type = "isothermal"
    if hasattr(fit_info, 'all_params'):
        pt = fit_info.all_params.get("profile_type")
        if pt is not None:
            profile_type = pt.best_guess if hasattr(pt, 'best_guess') else str(pt)

    # Which Madhusudhan & Seager parameterisation is the caller using?  The
    # Python Profile object takes T3 free and solves for P2; the older JAX
    # path took log_P2 free and derived T3.  Prefer the Python convention
    # whenever T3 is available, so the two paths agree.
    parametric_uses_T3 = (
        profile_type == "parametric"
        and _fit_info_value(fit_info, "T3", None) is not None)
    if profile_type == "parametric" and not parametric_uses_T3:
        if _fit_info_value(fit_info, "log_P2", None) is None:
            raise UnsupportedJAXFeatureError(
                "profile_type='parametric' requires either T3 (the "
                "TP_profile.set_parametric convention) or log_P2.")

    # ---- quenching setup -------------------------------------------------
    quench_enabled = False
    quench_species_mask = np.zeros(len(species_names), dtype=bool)
    if hasattr(fit_info, 'all_params'):
        log_pq = fit_info.all_params.get("log_P_quench")
        if log_pq is not None:
            pq_val = log_pq.best_guess if hasattr(log_pq, 'best_guess') else float(log_pq)
            if pq_val is not None and pq_val > -50:  # -99 means disabled
                quench_enabled = True
        qs = fit_info.all_params.get("quench_species")
        if qs is not None:
            qs_val = qs.best_guess if hasattr(qs, 'best_guess') else qs
            if qs_val is not None:
                for s in qs_val:
                    if s in species_names:
                        quench_species_mask[species_names.index(s)] = True
            else:
                # quench all species by default
                quench_species_mask[:] = True
        elif quench_enabled:
            quench_species_mask[:] = True

    # T_quench: explicit fit param or interpolate from T/P profile?
    fit_T_quench = "T_quench" in fit_info.fit_param_names

    # ---- dynamic stellar correction --------------------------------------
    fit_T_star = "T_star" in fit_info.fit_param_names
    fit_T_spot = "T_spot" in fit_info.fit_param_names
    fit_spot_cov_frac = "spot_cov_frac" in fit_info.fit_param_names
    dynamic_stellar = (
        T_star is not None and
        (fit_T_star or fit_T_spot or fit_spot_cov_frac))

    _f = np.float32  # store all arrays in forward-model precision
    data = {
        "T_grid":                jnp.array(T_grid, dtype=_f),
        "P_grid":                jnp.array(P_grid, dtype=_f),
        "log10_P_grid":          jnp.array(np.log10(P_grid), dtype=_f),
        "ln_P_grid":             jnp.array(ln_P_grid, dtype=_f),
        "lambda_grid":           jnp.array(lambda_grid, dtype=_f),
        "ln_scatt_lambda_ratio": jnp.array(np.log(1e-6 / lambda_grid), dtype=_f),
        "P_profile":             jnp.array(P_profile, dtype=_f),
        "ln_P_profile":          jnp.array(ln_P_profile, dtype=_f),
        "log10_P_profile":       jnp.array(np.log10(P_profile), dtype=_f),
        "ln_kBT_grid":           jnp.array(ln_kBT_grid, dtype=_f),
        "n_dens_grid":           jnp.array(n_dens_grid, dtype=_f),
        "n_dens_grid_scaled":    jnp.array(n_dens_grid * 1e-15, dtype=_f),
        "profile_p_lo":          jnp.array(profile_p_lo),
        "profile_p_hi":          jnp.array(profile_p_hi),
        "profile_p_frac":        jnp.array(profile_p_frac, dtype=_f),
        "logZs":                 jnp.array(logZs, dtype=_f),
        "CO_ratios":             jnp.array(CO_ratios, dtype=_f),
        "log_abundances":        jnp.array(log_abundances, dtype=_f),
        "all_masses":            jnp.array(all_masses, dtype=_f),
        "all_absorption_data":   jnp.array(all_absorption_data, dtype=_f),
        "opacity_species_indices": jnp.array(opacity_species_indices),
        "all_pol_sqr":           jnp.array(all_pol_sqr, dtype=_f),
        "all_coll_data":         jnp.array(all_coll_data, dtype=_f),
        "coll_indices":          jnp.array(coll_indices),
        "correction_factors":    jnp.array(correction_factors, dtype=_f),
        # Composite (spot-weighted) stellar spectrum on the trimmed grid.
        # The transit model only needs correction_factors, but the emission
        # model divides the planet flux by this directly.
        "stellar_spectrum":      jnp.array(stellar_spectrum, dtype=_f),
        "expanded_lambda_idx":   jnp.array(expanded_lambda_idx),
        "bin_ids":               jnp.array(bin_ids),
        "bin_weights_1d":        jnp.array(bin_weights_1d, dtype=_f),
        "n_bins":                n_bins,
        "ln_P_below":            jnp.array(np.log(P_below_arr), dtype=_f),
        "ln_P_above":            jnp.array(np.log(P_above_arr), dtype=_f),
        "offset_names":          offset_names,
        "offset_masks":          jnp.array(offset_masks, dtype=_f),
        "ref_pressure":          ref_pressure,
        "ch4_index":             ch4_index,
        "so2_index":             so2_index,
        "tio_index":             tio_index,
        "vo_index":              vo_index,
        "n_coll_pairs":          len(coll_data_list),
        # VMR / CLR
        "vmr_mode":              vmr_mode,
        "clr_mode":              clr_mode,
        "vmr_gas_names":         vmr_gas_names,
        "vmr_mapping":           vmr_mapping,
        "vmr_species_indices":   jnp.array(vmr_species_indices),
        "vmr_gas_indices":       jnp.array(vmr_gas_indices),
        "vmr_fracs":             jnp.array(vmr_fracs, dtype=_f),
        "vmr_gas_total_frac":    jnp.array(vmr_gas_total_frac, dtype=_f),
        "n_species":             len(species_names),
        "N_T":                   len(T_grid),
        "N_P":                   len(P_grid),
        # Python-side runtime validation bounds
        "min_temperature":       float(np.min(T_grid)),
        "max_temperature":       float(np.max(T_grid)),
        "logZ_min":              float(np.min(logZs)),
        "logZ_max":              float(np.max(logZs)),
        "CO_min":                float(np.min(CO_ratios)),
        "CO_max":                float(np.max(CO_ratios)),
        "P_min":                 float(np.min(P_grid)),
        "P_max":                 float(np.max(P_grid)),
        # dynamic stellar correction
        "dynamic_stellar":       dynamic_stellar,
        "fit_T_star":            fit_T_star,
        "default_T_star":        _f(T_star) if T_star is not None else _f(0.0),
        "stellar_blackbody":     bool(stellar_blackbody),
        # T/P profile type
        "profile_type":          profile_type,
        "parametric_uses_T3":    parametric_uses_T3,
        "expn2_x":               jnp.array(_EXPN2_X, dtype=_f),
        "expn2_y":               jnp.array(_EXPN2_Y, dtype=_f),
        # quenching
        "quench_enabled":        quench_enabled,
        "quench_species_mask":   jnp.array(quench_species_mask),
        "fit_T_quench":          fit_T_quench,
        # H- absorption
        "add_H_minus":           add_H_minus,
        "h_minus_cross":         jnp.array(h_minus_cross, dtype=_f) if add_H_minus else jnp.zeros((0, 0), dtype=_f),
        "el_index":              el_index,
        "H_index":               H_index,
        # model mode flags
        "limb_asym":             _get_fit_info_bool(fit_info, "limb_asym"),
        "patchy":                _get_fit_info_bool(fit_info, "cloud_cov_frac", invert=True),
        "h2s_idx":               int(species_names.index("H2S")) if "H2S" in species_names else -1,
        "cs2_idx":               int(species_names.index("CS2")) if "CS2" in species_names else -1,
        "sulfur_transition_mode": (
            hasattr(fit_info, 'all_params') and
            "log_H2S_vmr" in fit_info.all_params and
            fit_info.all_params["log_H2S_vmr"].best_guess is not None
        ),
        # photochem mode
        "photochem_mode":        photochem_mode,
        "pc_abund_base":         jnp.array(pc_abund_stack, dtype=_f),
        "pc_T_on_profile":       jnp.array(pc_T_on_profile, dtype=_f),
        "pc_mol_indices":        tuple(int(x) for x in pc_mol_indices),
        "pc_mol_names":          pc_mols,
        "pc_h2_idx":             int(pc_h2_idx),
        "pc_he_idx":             int(pc_he_idx),
        "pc_n_mols":             len(pc_mols),
    }

    if dynamic_stellar:
        stellar_temps = _to_numpy(atm_solver.stellar_spectra_temps)
        stellar_grid = _to_numpy(atm_solver.stellar_spectra)
        unspotted, _ = atm_solver.get_stellar_spectrum(
            T_star, T_star, 0, stellar_blackbody)
        data["stellar_spectra_temps"] = jnp.array(stellar_temps, dtype=_f)
        data["stellar_spectra_grid"]  = jnp.array(stellar_grid[:, trim_lo:trim_hi], dtype=_f)
        data["unspotted_spectrum"]    = jnp.array(
            _to_numpy(unspotted)[trim_lo:trim_hi], dtype=_f)

    return data


# ---------------------------------------------------------------------------
# JAX forward model helpers
# ---------------------------------------------------------------------------

def _jax_get_abundances(logZ, CO_ratio, log_abundances, logZs, CO_ratios):
    """Interpolate abundance tables → (n_species, N_T, N_P)."""
    # log_abundances: (n_logZ, n_CO, n_species, N_T, N_P)
    interp = jax_regular_grid_interp(
        logZs, CO_ratios, log_abundances,
        jnp.atleast_1d(logZ), jnp.atleast_1d(CO_ratio))
    return _FM_DTYPE(10.0) ** interp[0]


def _jax_interp_abundances_to_profile(abundances, all_masses,
                                      T_grid, P_grid,
                                      T_profile, P_profile,
                                      profile_p_lo=None,
                                      profile_p_hi=None,
                                      profile_p_frac=None):
    """Interpolate abundances from (T_grid, P_grid) → atmospheric layers.

    Returns (atm_abundances (NUM_LAYERS, n_species), mu_profile (NUM_LAYERS,)).
    Mimics the original log-space interpolation.
    """
    # abundances: (n_species, N_T, N_P)  → transpose to (N_T, N_P, n_species)
    abundances_tp = abundances.transpose(1, 2, 0)
    log_abund = jnp.where(
        abundances_tp > 0,
        jnp.log10(abundances_tp),
        _LOG10_MIN_ABUNDANCE)

    t_idx = jnp.interp(T_profile, T_grid,
                       jnp.arange(len(T_grid), dtype=_FM_DTYPE))
    t_lo = jnp.floor(t_idx).astype(int)
    t_hi = jnp.minimum(t_lo + 1, len(T_grid) - 1)
    t_frac = (t_idx - t_lo)[:, None]

    if profile_p_lo is None:
        log10_P = jnp.log10(P_grid)
        p_idx = jnp.interp(jnp.log10(P_profile), log10_P,
                           jnp.arange(len(log10_P), dtype=_FM_DTYPE))
        profile_p_lo = jnp.floor(p_idx).astype(int)
        profile_p_hi = jnp.minimum(profile_p_lo + 1, len(log10_P) - 1)
        profile_p_frac = p_idx - profile_p_lo
    p_frac = profile_p_frac[:, None]

    log_abund_atm = (
        log_abund[t_lo, profile_p_lo] * (1 - t_frac) * (1 - p_frac) +
        log_abund[t_hi, profile_p_lo] * t_frac * (1 - p_frac) +
        log_abund[t_lo, profile_p_hi] * (1 - t_frac) * p_frac +
        log_abund[t_hi, profile_p_hi] * t_frac * p_frac
    )

    atm_abund = _FM_DTYPE(10.0) ** log_abund_atm
    mu_profile = jnp.sum(atm_abund * all_masses[None, :], axis=1)
    return atm_abund, mu_profile


def _jax_compute_absorption_coeff(abundances,
                                  all_absorption_data, opacity_species_indices,
                                  all_pol_sqr,
                                  all_coll_data, coll_indices, n_coll_pairs,
                                  T_grid, P_grid, lambda_grid,
                                  scatt_factor, scatt_slope):
    """Total absorption coefficient on (N_T, N_P, N_lambda)."""
    # --- gas absorption ---
    opacity_abund = abundances[opacity_species_indices]        # (n_op, N_T, N_P)
    absorption_coeff = jnp.sum(
        opacity_abund[:, :, :, None] * all_absorption_data, axis=0)

    # --- Rayleigh scattering ---
    # Match the NumPy path exactly:
    #   rayleigh = C * scatt_factor * n_dens * sum_pol
    #              * scatt_ref^(slope-4) / lambda^slope
    # Evaluate the wavelength dependence as one combined exponent so we do
    # not accidentally introduce extra powers of the reference wavelength.
    sum_pol = jnp.sum(abundances * all_pol_sqr[:, None, None], axis=0)  # (N_T, N_P)
    n_dens = P_grid[None, :] / (_k_B * T_grid[:, None])
    scatt_log_scale = (
        scatt_slope * jnp.log(_SCATT_REF / lambda_grid[None, None, :])
        - _FM_DTYPE(4.0) * _LN_SCATT_REF
    )
    rayleigh = (scatt_factor * _RAYLEIGH_CONST
                * _POL_SQR_INV_SCALE
                * (n_dens * sum_pol)[:, :, None]
                * jnp.exp(scatt_log_scale))
    absorption_coeff = absorption_coeff + rayleigh

    # --- collisional absorption ---
    # n1*n2 can reach ~1e57 at high P, overflowing float32 (max 3.4e38).
    # Rescale: split n_dens into (n_dens * s) with s chosen so the product
    # (n_dens*s)^2 stays in range.  s = 1e-15 → max (n_dens*s) ~ 7e13,
    # max (n_dens*s)^2 ~ 5e27.  Compensate by pre-scaling coll_abs by s^-2.
    _n_scale = _FM_DTYPE(1e-15)
    n_dens_s = P_grid[None, :] / (_k_B * T_grid[:, None]) * _n_scale
    for p in range(n_coll_pairs):
        idx1 = coll_indices[p, 0]
        idx2 = coll_indices[p, 1]
        n1s = abundances[idx1] * n_dens_s
        n2s = abundances[idx2] * n_dens_s
        # coll_abs is ~1e-55, times 1/s^2 = 1e30 → ~1e-25, safe in float32
        coll_abs_scaled = all_coll_data[p] / (_n_scale * _n_scale)
        absorption_coeff = absorption_coeff + (
            coll_abs_scaled[:, None, :] * (n1s * n2s)[:, :, None])

    return absorption_coeff


def _jax_interpolate_to_profile(absorption_coeff,
                                T_grid, P_grid, T_profile, P_profile,
                                profile_p_lo=None,
                                profile_p_hi=None,
                                profile_p_frac=None,
                                ln_P_profile=None,
                                ln_P_grid=None,
                                ln_kBT_grid=None):
    """Interpolate absorption from (T_grid, P_grid) → atmospheric layers.

    Returns (NUM_LAYERS, N_lambda).

    Supports both isothermal and non-isothermal T/P profiles via full
    bilinear interpolation in (1/T, ln P) space.
    """
    if ln_P_grid is None:
        ln_P_grid = jnp.log(P_grid)
    if ln_kBT_grid is None or ln_kBT_grid.shape[0] != T_grid.shape[0]:
        ln_kBT_grid = jnp.log(_k_B * T_grid)

    inv_T = 1.0 / T_grid[::-1]
    ln_n_grid = (ln_P_grid[None, :] - ln_kBT_grid[::-1, None])[:, :, None]
    ln_absorption = jnp.where(
        absorption_coeff[::-1] > 0,
        jnp.log(absorption_coeff[::-1]),
        _FM_DTYPE(0.0))
    ln_cross = jnp.where(
        absorption_coeff[::-1] > 0,
        ln_absorption - ln_n_grid,
        _LN_MIN_CROSS_SEC)
    ln_cross = jnp.maximum(ln_cross, _LN_MIN_CROSS_SEC)  # (N_T, N_P, N_lambda)

    # Full bilinear interpolation for each atmospheric layer
    inv_T_profile = 1.0 / T_profile
    target_ln_P = ln_P_profile if ln_P_profile is not None else jnp.log(P_profile)

    t_idx = jnp.interp(inv_T_profile, inv_T, jnp.arange(len(inv_T), dtype=_FM_DTYPE))
    t_lo = jnp.floor(t_idx).astype(int)
    t_hi = jnp.minimum(t_lo + 1, len(inv_T) - 1)
    t_frac = t_idx - t_lo

    if profile_p_lo is None:
        p_idx = jnp.interp(target_ln_P, ln_P_grid,
                           jnp.arange(len(ln_P_grid), dtype=_FM_DTYPE))
        p_lo = jnp.floor(p_idx).astype(int)
        p_hi = jnp.minimum(p_lo + 1, len(ln_P_grid) - 1)
        p_frac = p_idx - p_lo
    else:
        p_lo = profile_p_lo
        p_hi = profile_p_hi
        p_frac = profile_p_frac

    # Bilinear: interpolate ln(cross_sec) at each layer's (T, P)
    # Shape broadcasts: t_frac/p_frac are (NUM_LAYERS,), ln_cross indexed
    # gives (NUM_LAYERS, N_lambda)
    t_frac = t_frac[:, None]
    p_frac = p_frac[:, None]
    ln_cross_atm = (
        ln_cross[t_lo, p_lo] * (1 - t_frac) * (1 - p_frac) +
        ln_cross[t_hi, p_lo] * t_frac * (1 - p_frac) +
        ln_cross[t_lo, p_hi] * (1 - t_frac) * p_frac +
        ln_cross[t_hi, p_hi] * t_frac * p_frac
    )

    # Convert back: absorption = cross_sec * n_dens = exp(ln_cross) * P/(k_B*T).
    # In float32, computing P/(k_B*T) directly causes overflow in the backward
    # pass because (k_B*T)^2 ~ 1e-41 underflows.  Absorb n_dens into the exp
    # to keep the backward pass in log-space where gradients are O(1/T).
    ln_n_atm = target_ln_P - jnp.log(_k_B * T_profile)
    return jnp.exp(ln_cross_atm + ln_n_atm[:, None])


def _jax_interpolate_to_profile_isothermal(absorption_coeff,
                                           T_grid, P_grid,
                                           T_profile, P_profile,
                                           profile_p_lo=None,
                                           profile_p_hi=None,
                                           profile_p_frac=None,
                                           ln_P_profile=None,
                                           ln_P_grid=None,
                                           ln_kBT_grid=None):
    """Fast interpolation path for isothermal profiles."""
    if ln_P_grid is None:
        ln_P_grid = jnp.log(P_grid)
    if ln_kBT_grid is None or ln_kBT_grid.shape[0] != T_grid.shape[0]:
        ln_kBT_grid = jnp.log(_k_B * T_grid)

    inv_T = 1.0 / T_grid[::-1]
    ln_n_grid = (ln_P_grid[None, :] - ln_kBT_grid[::-1, None])[:, :, None]
    ln_absorption = jnp.where(
        absorption_coeff[::-1] > 0,
        jnp.log(absorption_coeff[::-1]),
        _FM_DTYPE(0.0))
    ln_cross = jnp.where(
        absorption_coeff[::-1] > 0,
        ln_absorption - ln_n_grid,
        _LN_MIN_CROSS_SEC)
    ln_cross = jnp.maximum(ln_cross, _LN_MIN_CROSS_SEC)

    T_val = T_profile[0]
    inv_T_val = 1.0 / T_val
    t_idx = jnp.interp(inv_T_val, inv_T, jnp.arange(len(inv_T), dtype=_FM_DTYPE))
    t_lo = jnp.floor(t_idx).astype(int)
    t_hi = jnp.minimum(t_lo + 1, len(inv_T) - 1)
    t_frac = t_idx - t_lo
    ln_cross_at_T = ln_cross[t_lo] * (1 - t_frac) + ln_cross[t_hi] * t_frac

    target_ln_P = ln_P_profile if ln_P_profile is not None else jnp.log(P_profile)
    if profile_p_lo is None:
        p_idx = jnp.interp(target_ln_P, ln_P_grid,
                           jnp.arange(len(ln_P_grid), dtype=_FM_DTYPE))
        p_lo = jnp.floor(p_idx).astype(int)
        p_hi = jnp.minimum(p_lo + 1, len(ln_P_grid) - 1)
        p_frac = p_idx - p_lo
    else:
        p_lo = profile_p_lo
        p_hi = profile_p_hi
        p_frac = profile_p_frac
    p_frac = p_frac[:, None]
    ln_cross_atm = ln_cross_at_T[p_lo] * (1 - p_frac) + ln_cross_at_T[p_hi] * p_frac

    ln_n_atm = target_ln_P - jnp.log(_k_B * T_profile)
    return jnp.exp(ln_cross_atm + ln_n_atm[:, None])


# ---------------------------------------------------------------------------
# Parametric T/P profile (Madhusudhan & Seager 2009)
# ---------------------------------------------------------------------------

def _jax_parametric_profile_from_ln_pressure(T0, log_P1, alpha1, alpha2,
                                             log_P2, log_P3, ln_P_profile):
    """Compute parametric T/P profile from a precomputed ln-pressure grid."""
    log_P0 = ln_P_profile[0]
    ln_P1 = log_P1 * _LN_10
    ln_P2 = log_P2 * _LN_10
    ln_P3 = log_P3 * _LN_10

    alpha1 = jnp.maximum(alpha1, _FM_DTYPE(0.01))
    alpha2 = jnp.maximum(alpha2, _FM_DTYPE(0.01))

    T1 = T0 + ((ln_P1 - log_P0) / alpha1) ** 2
    T2 = T1 - ((ln_P1 - ln_P2) / alpha2) ** 2
    T3 = T2 + ((ln_P3 - ln_P2) / alpha2) ** 2

    T_region1 = T0 + ((ln_P_profile - log_P0) / alpha1) ** 2
    T_region2 = T2 + ((ln_P_profile - ln_P2) / alpha2) ** 2
    T_region3 = jnp.full_like(ln_P_profile, T3)

    # Match TP_profile.set_parametric() exactly. The legacy Python path writes
    # region1, then region2, then region3 with masked assignment, so region3
    # takes precedence if the retrieved pressures overlap non-monotonically
    # (e.g. log_P1 > log_P3). Nested where(region1, ...) does not preserve
    # that overwrite order.
    return jnp.where(
        ln_P_profile >= ln_P3, T_region3,
        jnp.where(ln_P_profile >= ln_P1, T_region2, T_region1))


def _jax_parametric_profile_T3_from_ln_pressure(T0, log_P1, alpha1, alpha2,
                                                log_P3, T3, ln_P_profile):
    """JAX port of TP_profile.Profile.set_parametric().

    This is the parameterisation the Python ``Profile`` object actually uses
    (Madhusudhan & Seager 2009 with ``T3`` free and ``P2`` solved for), as
    opposed to :func:`_jax_parametric_profile_from_ln_pressure`, which takes
    ``log_P2`` free and derives ``T3``.  ``log_P1`` / ``log_P3`` are log10 of
    the pressure in Pa, matching ``params_dict``.
    """
    ln_P0 = ln_P_profile[0]
    ln_P1 = log_P1 * _LN_10
    ln_P3 = log_P3 * _LN_10

    alpha1 = jnp.maximum(alpha1, _FM_DTYPE(0.01))
    alpha2 = jnp.maximum(alpha2, _FM_DTYPE(0.01))

    # Solve for P2 exactly as set_parametric() does.
    ln_P2 = (alpha2 ** 2 * (T0 + (ln_P1 - ln_P0) ** 2 / alpha1 ** 2 - T3)
             - ln_P1 ** 2 + ln_P3 ** 2) / (2 * (ln_P3 - ln_P1))
    T2 = T3 - (ln_P3 - ln_P2) ** 2 / alpha2 ** 2

    T_region1 = T0 + (ln_P_profile - ln_P0) ** 2 / alpha1 ** 2
    T_region2 = T2 + (ln_P_profile - ln_P2) ** 2 / alpha2 ** 2

    # Matches the if / elif / else ordering of set_parametric().
    return jnp.where(
        ln_P_profile < ln_P1, T_region1,
        jnp.where(ln_P_profile < ln_P3, T_region2,
                  jnp.full_like(ln_P_profile, T3)))


def _jax_expn2(x, expn2_x, expn2_y):
    """Table-interpolated E_2(x), used by the radiative-solution profile."""
    return jnp.interp(x, expn2_x, expn2_y,
                      left=_FM_DTYPE(1.0), right=_FM_DTYPE(0.0))


def _jax_radiative_solution_profile(params_dict, data):
    """JAX port of TP_profile.Profile.set_from_radiative_solution().

    Line et al. (2013), Eqs. 13-16.  Requires the ``expn2_x`` / ``expn2_y``
    lookup tables to be present in ``data`` (added by ``prepare_jax_data``).
    """
    P_profile = data["P_profile"]
    T_star = _FM_DTYPE(params_dict["T_star"])
    Rs = _FM_DTYPE(params_dict["Rs"])
    a = _FM_DTYPE(params_dict["a"])
    Mp = _FM_DTYPE(params_dict["Mp"])
    Rp = _FM_DTYPE(params_dict["Rp"])
    beta = _FM_DTYPE(params_dict["beta"])
    k_th = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_k_th"])
    gamma = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_gamma"])
    alpha = _FM_DTYPE(params_dict.get("alpha", 0.0))
    T_int = _FM_DTYPE(params_dict.get("T_int", 100.0))

    g = _G * Mp / Rp ** 2
    T_eq = beta * jnp.sqrt(Rs / (2 * a)) * T_star
    taus = k_th * P_profile / g

    expn2_x = data["expn2_x"]
    expn2_y = data["expn2_y"]

    def incoming_stream_contribution(gam):
        gt = gam * taus
        return (_FM_DTYPE(0.75) * T_eq ** 4
                * (_FM_DTYPE(2.0 / 3)
                   + 2.0 / (3 * gam) * (1 + (gt / 2 - 1) * jnp.exp(-gt))
                   + 2.0 * gam / 3 * (1 - taus ** 2 / 2)
                   * _jax_expn2(gt, expn2_x, expn2_y)))

    e1 = incoming_stream_contribution(gamma)
    T4 = _FM_DTYPE(0.75) * T_int ** 4 * (_FM_DTYPE(2.0 / 3) + taus) + (1 - alpha) * e1

    log_gamma2 = params_dict.get("log_gamma2")
    if log_gamma2 is not None:
        gamma2 = _FM_DTYPE(10.0) ** _FM_DTYPE(log_gamma2)
        T4 = T4 + alpha * incoming_stream_contribution(gamma2)

    return jnp.maximum(T4, _FM_DTYPE(0.0)) ** _FM_DTYPE(0.25)


def _jax_parametric_profile(T0, log_P1, alpha1, alpha2, log_P2, log_P3,
                            P_profile):
    """Compute parametric T/P profile in pure JAX (JIT-compatible).

    Parameters match TP_profile.set_parametric() but log_P1/P2/P3 are
    log10(pressure in Pa) rather than pressure in Pa.

    Returns T_profile (NUM_LAYERS,).
    """
    return _jax_parametric_profile_from_ln_pressure(
        T0, log_P1, alpha1, alpha2, log_P2, log_P3, jnp.log(P_profile))


# ---------------------------------------------------------------------------
# Chemical quenching
# ---------------------------------------------------------------------------

def _jax_apply_quenching(abundances, T_quench, P_quench,
                         quench_species_mask, T_grid, log10_P_grid):
    """Freeze abundances of quenched species at pressures <= P_quench.

    abundances: (n_species, N_T, N_P)
    quench_species_mask: (n_species,) boolean array
    T_quench: scalar temperature at quench point
    P_quench: scalar pressure at quench point

    Returns modified abundances (n_species, N_T, N_P).
    """
    log_abund = jnp.where(
        abundances > 0,
        jnp.log10(abundances),
        _LOG10_MIN_ABUNDANCE)

    # Interpolate each species abundance at (T_quench, P_quench)
    # log_abund shape: (n_species, N_T, N_P) -> need per-species interp
    # Transpose to (N_T, N_P, n_species) for regular_grid_interp
    log_abund_tp = log_abund.transpose(1, 2, 0)  # (N_T, N_P, n_species)
    quench_log_abund = jax_regular_grid_interp(
        T_grid, log10_P_grid, log_abund_tp,
        jnp.atleast_1d(T_quench),
        jnp.atleast_1d(jnp.log(P_quench) / _LN_10)
    )[0]  # (n_species,)
    quench_abund = _FM_DTYPE(10.0) ** quench_log_abund  # (n_species,)

    # For each species where quench_species_mask is True,
    # set abundance to quench value at all P <= P_quench
    P_mask = (log10_P_grid <= (jnp.log(P_quench) / _LN_10))  # (N_P,) boolean
    # Broadcast quench abundances without materializing a full dense tensor.
    quench_vals = quench_abund[:, None, None]
    # Apply: where mask is True AND species is quenched, use quench value
    species_mask = quench_species_mask[:, None, None]  # (n_species, 1, 1)
    P_mask_3d = P_mask[None, None, :]  # (1, 1, N_P)
    should_quench = species_mask & P_mask_3d

    return jnp.where(should_quench, quench_vals, abundances)


def _jax_get_temperature_profile(params_dict, data):
    """Return the atmosphere temperature profile for this parameter set."""
    # Photochem mode: pre-baked T/P profile shifted by delta_T
    if data.get("photochem_mode", False):
        delta_T = _FM_DTYPE(params_dict.get("delta_T", 0.0))
        return data["pc_T_on_profile"] + delta_T

    profile_type = data.get("profile_type", "isothermal")
    if profile_type == "parametric":
        if data.get("parametric_uses_T3", False):
            # Same parameterisation as TP_profile.Profile.set_parametric()
            # (T3 free, P2 solved for), which is what the Python reference
            # path uses.  Keeps the JAX and NumPy models consistent.
            return _jax_parametric_profile_T3_from_ln_pressure(
                _FM_DTYPE(params_dict["T0"]),
                _FM_DTYPE(params_dict["log_P1"]),
                _FM_DTYPE(params_dict["alpha1"]),
                _FM_DTYPE(params_dict["alpha2"]),
                _FM_DTYPE(params_dict["log_P3"]),
                _FM_DTYPE(params_dict["T3"]),
                data["ln_P_profile"])
        return _jax_parametric_profile_from_ln_pressure(
            _FM_DTYPE(params_dict["T0"]),
            _FM_DTYPE(params_dict["log_P1"]),
            _FM_DTYPE(params_dict["alpha1"]),
            _FM_DTYPE(params_dict["alpha2"]),
            _FM_DTYPE(params_dict["log_P2"]),
            _FM_DTYPE(params_dict["log_P3"]),
            data["ln_P_profile"])

    if profile_type == "radiative_solution":
        return _jax_radiative_solution_profile(params_dict, data)

    if profile_type == "external":
        # Caller supplies the temperature at every layer of data["P_profile"]
        # directly.  Used by jax_eclipse_depth_calculator for arbitrary
        # Profile objects.
        return jnp.asarray(params_dict["T_profile"], dtype=_FM_DTYPE)

    T = _FM_DTYPE(params_dict["T"])
    return jnp.ones(NUM_LAYERS, dtype=_FM_DTYPE) * T


def _jax_get_master_grid_abundances(params_dict, data, T_profile):
    """Return abundances on the static (T_grid, P_grid) master grid."""
    T_grid = data["T_grid"]
    P_grid = data["P_grid"]

    if data.get("photochem_mode", False):
        # Start from pre-baked photochem base (n_species, N_T, N_P)
        abundances = data["pc_abund_base"]

        # Apply log10_mult scaling per molecule — runs in JAX, JIT-compiled
        for i, mol in enumerate(data["pc_mol_names"]):
            if mol in ("H2S", "CS2"):   # handled by log10_mult_S below
                continue
            idx = int(data["pc_mol_indices"][i])
            if idx < 0:
                continue
            key = f"log10_mult_{mol}"
            if key in params_dict:
                scale = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict[key])
                scaled = jnp.clip(abundances[idx] * scale, _EPS, _FM_DTYPE(1.0))
                abundances = abundances.at[idx].set(scaled)
        if "log10_mult_S" in params_dict:
            scale_S = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log10_mult_S"])
            for mol, idx_key in [("H2S", "h2s_idx"), ("CS2", "cs2_idx")]:
                idx = data.get(idx_key, -1)
                if idx >= 0:
                    scaled = jnp.clip(abundances[idx] * scale_S, _EPS, _FM_DTYPE(1.0))
                    abundances = abundances.at[idx].set(scaled)
                    
        # Recompute H2/He so VMRs still sum to 1
        h2_idx = data["pc_h2_idx"]
        he_idx = data["pc_he_idx"]
        if h2_idx >= 0 and he_idx >= 0:
            bg_mask = jnp.ones(data["n_species"], dtype=_FM_DTYPE)
            bg_mask = bg_mask.at[h2_idx].set(_FM_DTYPE(0.0))
            bg_mask = bg_mask.at[he_idx].set(_FM_DTYPE(0.0))
            sum_minor = jnp.einsum("s,stp->tp", bg_mask, abundances)
            remainder = jnp.clip(_FM_DTYPE(1.0) - sum_minor, _EPS, _FM_DTYPE(1.0))
            abundances = abundances.at[h2_idx].set(remainder * _FM_DTYPE(0.836))
            abundances = abundances.at[he_idx].set(remainder * _FM_DTYPE(0.164))

        return abundances

    if data["vmr_mode"]:
        gas_names = data["vmr_gas_names"]
        log_vmrs = jnp.array(
            [_FM_DTYPE(params_dict[f"log_{g}"]) for g in gas_names[:-1]])
        vmrs_main = _FM_DTYPE(10.0) ** log_vmrs
        # Multiply by per-gas total mapping fraction so the bg subtracts
        # the *actual* molecular VMR consumed (matters for the tied 'S'
        # gas, where 10**log_S is elemental S and only 2/3 of it shows up
        # as molecules; for ordinary gases and 'H2-He' this multiplier is
        # 1.0 and the bg calc is unchanged).
        eff_vmrs_main = vmrs_main * data["vmr_gas_total_frac"][:-1]
        bkg_vmr = jnp.maximum(_FM_DTYPE(1.0) - jnp.sum(eff_vmrs_main), _FM_DTYPE(1e-99))
        all_vmrs = jnp.concatenate(
            [vmrs_main, jnp.array([bkg_vmr], dtype=_FM_DTYPE)])

        abundances = jnp.full(
            (data["n_species"], data["N_T"], data["N_P"]), _FM_DTYPE(1e-99),
            dtype=_FM_DTYPE)
        mapped_vmrs = jnp.maximum(
            all_vmrs[data["vmr_gas_indices"]] * data["vmr_fracs"], _FM_DTYPE(1e-99))
        abundances = abundances.at[data["vmr_species_indices"]].add(
            mapped_vmrs[:, None, None])

    elif data["clr_mode"]:
        gas_names = data["vmr_gas_names"]
        clrs = jnp.array(
            [_FM_DTYPE(params_dict[f"clr_{g}"]) for g in gas_names[:-1]])
        geo_mean = jnp.exp(jnp.mean(clrs))
        clrs_with_bkg = jnp.concatenate(
            [clrs, jnp.array([_FM_DTYPE(0.0)])])
        all_vmrs = jnp.exp(clrs_with_bkg) * geo_mean

        abundances = jnp.full(
            (data["n_species"], data["N_T"], data["N_P"]), _FM_DTYPE(1e-99),
            dtype=_FM_DTYPE)
        mapped_vmrs = jnp.maximum(
            all_vmrs[data["vmr_gas_indices"]] * data["vmr_fracs"], _FM_DTYPE(1e-99))
        abundances = abundances.at[data["vmr_species_indices"]].add(
            mapped_vmrs[:, None, None])

    else:
        logZ = params_dict["logZ"]
        CO_ratio = params_dict["CO_ratio"]
        base_abundances = _jax_get_abundances(
            logZ, CO_ratio,
            data["log_abundances"], data["logZs"], data["CO_ratios"])
        # log_abundances only covers the base equilibrium species; pad to
        # n_species so the appended species (DMS, CS2, CH3SH) get a slot and
        # all_masses / atm_abund shapes stay consistent.
        n_base = base_abundances.shape[0]
        n_extra = data["n_species"] - n_base
        if n_extra > 0:
            extra = jnp.full(
                (n_extra, data["N_T"], data["N_P"]), _FM_DTYPE(1e-99),
                dtype=_FM_DTYPE)
            abundances = jnp.concatenate([base_abundances, extra], axis=0)
        else:
            abundances = base_abundances

        ch4_mult = 10.0 ** params_dict.get("log_CH4_mult", 0.0)
        ch4_idx = data["ch4_index"]
        if ch4_idx >= 0:
            abundances = abundances.at[ch4_idx].multiply(ch4_mult)

        for key, idx_name in [("log_SO2", "so2_index"), ("log_CH4", "ch4_index"),
                              ("log_TiO", "tio_index"), ("log_VO", "vo_index"),
                              ("log_CS2", "cs2_idx")]:
            if key in params_dict and params_dict[key] is not None:
                idx = data[idx_name]
                if idx >= 0:
                    override_val = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict[key])
                    abundances = abundances.at[idx].set(
                        jnp.full((data["N_T"], data["N_P"]), override_val,
                                 dtype=_FM_DTYPE))

        # ---- tied sulfur free retrieval (toggleable) ----------------------
        # When `log_S` is present in params_dict (i.e. set or fit by the
        # user via fit_info), 10**log_S is interpreted as the ELEMENTAL
        # sulfur abundance (S atoms per total atmospheric molecules). The
        # S-atom budget is split 1/3 into H2S (1 S atom/molecule) and 2/3
        # into CS2 (2 S atoms/molecule), which translates into equal
        # molecular VMRs:
        #     VMR(H2S) = (1/3) * 10**log_S   (carries 1/3 of S atoms)
        #     VMR(CS2) = (1/3) * 10**log_S   (carries 2/3 of S atoms via 2 S/mol)
        # So elemental S budget = 1*VMR(H2S) + 2*VMR(CS2) = 10**log_S,
        # directly comparable to S/H once divided by VMR(H).
        # Toggle: just include `log_S` in your fit (or set a fixed value);
        # otherwise it is absent from params_dict and this branch is dead
        # at JIT trace time (zero runtime cost when disabled). This override
        # is equilibrium-chemistry mode only, matching the log_SO2/log_CH4/
        # log_TiO/log_VO pattern above. For a parallel toggle in fit_vmr
        # mode, list "S" in fit_info.gases (see prepare_jax_data).
        if "log_S" in params_dict and params_dict["log_S"] is not None:
            total_S = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_S"])
            h2s_idx = data["h2s_idx"]
            cs2_idx = data["cs2_idx"]
            split_val = total_S * _FM_DTYPE(1.0 / 3.0)
            if h2s_idx >= 0:
                abundances = abundances.at[h2s_idx].set(
                    jnp.full((data["N_T"], data["N_P"]), split_val,
                             dtype=_FM_DTYPE))
            if cs2_idx >= 0:
                abundances = abundances.at[cs2_idx].set(
                    jnp.full((data["N_T"], data["N_P"]), split_val,
                             dtype=_FM_DTYPE))

    if data.get("quench_enabled", False):
        log_P_quench = _FM_DTYPE(params_dict.get("log_P_quench", -99.0))
        ln_P_quench = log_P_quench * _LN_10
        P_quench = jnp.exp(ln_P_quench)
        if data.get("fit_T_quench", False):
            T_quench = _FM_DTYPE(params_dict["T_quench"])
        else:
            T_quench = jnp.interp(ln_P_quench, data["ln_P_profile"], T_profile)
        abundances = _jax_apply_quenching(
            abundances, T_quench, P_quench,
            data["quench_species_mask"], T_grid, data["log10_P_grid"])

    if data.get("sulfur_transition_mode", False):
        log10_Pt = _FM_DTYPE(params_dict["log10_P_S_transition"])
        Pt = _FM_DTYPE(10.0) ** log10_Pt
        P_grid = data["P_grid"]
        h2s_vmr = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_H2S_vmr"])
        cs2_vmr = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_CS2_vmr"])
        # H2S present at P > Pt (deep atm), CS2 at P < Pt (upper atm)
        h2s_row = jnp.where(P_grid > Pt, h2s_vmr, _EPS)   # (N_P,)
        cs2_row = jnp.where(P_grid < Pt, cs2_vmr, _EPS)   # (N_P,)
        h2s_2d = jnp.broadcast_to(h2s_row[None, :], (data["N_T"], data["N_P"]))
        cs2_2d = jnp.broadcast_to(cs2_row[None, :], (data["N_T"], data["N_P"]))
        h2s_idx = data["h2s_idx"]
        cs2_idx = data["cs2_idx"]
        if h2s_idx >= 0:
            abundances = abundances.at[h2s_idx].set(h2s_2d)
        if cs2_idx >= 0:
            abundances = abundances.at[cs2_idx].set(cs2_2d)

    return abundances


def _jax_build_atmosphere_state(params_dict, data):
    """Build shared atmosphere state that both clear and cloudy paths use."""
    with jax.named_scope("build_atmosphere_state"):
        Rp = _FM_DTYPE(params_dict["Rp"])
        Rs = _FM_DTYPE(params_dict["Rs"])
        Mp = _FM_DTYPE(params_dict["Mp"])
        T_grid = data["T_grid"]
        P_grid = data["P_grid"]
        P_profile = data["P_profile"]
        ln_P_profile = data["ln_P_profile"]

        with jax.named_scope("temperature_profile"):
            T_profile = _jax_get_temperature_profile(params_dict, data)
        with jax.named_scope("master_grid_abundances"):
            abundances = _jax_get_master_grid_abundances(params_dict, data, T_profile)
        with jax.named_scope("mu_profile"):
            _, mu_profile = _jax_interp_abundances_to_profile(
                abundances, data["all_masses"], T_grid, P_grid, T_profile, P_profile,
                data["profile_p_lo"], data["profile_p_hi"], data["profile_p_frac"])
        with jax.named_scope("bound_check"):
            T_star = params_dict.get("T_star")
            is_unbound = _jax_atmosphere_is_unbound(
                T_profile, mu_profile, Mp, Rp, Rs,
                None if T_star is None else _FM_DTYPE(T_star))
        with jax.named_scope("hydrostatic_solve"):
            def solve_bound(_):
                return _jax_hydrostatic_solve(
                    ln_P_profile, T_profile, mu_profile, Mp, Rp,
                    data["ln_P_below"], data["ln_P_above"])

            def solve_unbound(_):
                return (jnp.full_like(P_profile, jnp.nan),
                        jnp.full((P_profile.shape[0] - 1,), jnp.nan, dtype=_FM_DTYPE))

            radii, dr = jax.lax.cond(is_unbound, solve_unbound, solve_bound, operand=None)

        return Rs, T_grid, P_grid, P_profile, T_profile, abundances, radii, dr, is_unbound


def _jax_get_temperature_slab(T_profile, T_grid, disable_fast_t=False):
    """Return an active temperature slab and whether the fast path is usable."""
    if disable_fast_t:
        return False, jnp.int32(0)
    t_min = jnp.min(T_profile)
    t_max = jnp.max(T_profile)
    t_start = jnp.maximum(jnp.searchsorted(T_grid, t_min, side="left") - 1, 0)
    t_end = jnp.minimum(
        jnp.searchsorted(T_grid, t_max, side="left") + 1, T_grid.shape[0])
    span = t_end - t_start
    t0 = jnp.minimum(t_start, T_grid.shape[0] - _FAST_T_GRID_SPAN)
    return span <= _FAST_T_GRID_SPAN, t0.astype(jnp.int32)


def _jax_slice_temperature_views(abundances, data, t_start):
    """Slice the temperature dimension down to the fixed fast-path slab."""
    n_species = data["n_species"]
    n_p = data["N_P"]
    n_lambda = data["lambda_grid"].shape[0]
    n_opacity_species = data["all_absorption_data"].shape[0]
    n_coll_pairs = data["all_coll_data"].shape[0]

    _i0 = jnp.int32(0)
    T_grid = jax.lax.dynamic_slice(
        data["T_grid"], (t_start,), (_FAST_T_GRID_SPAN,))
    abundances = jax.lax.dynamic_slice(
        abundances, (_i0, t_start, _i0), (n_species, _FAST_T_GRID_SPAN, n_p))
    all_absorption_data = jax.lax.dynamic_slice(
        data["all_absorption_data"], (_i0, t_start, _i0, _i0),
        (n_opacity_species, _FAST_T_GRID_SPAN, n_p, n_lambda))
    all_coll_data = jax.lax.dynamic_slice(
        data["all_coll_data"], (_i0, t_start, _i0),
        (n_coll_pairs, _FAST_T_GRID_SPAN, n_lambda))
    n_dens_grid = jax.lax.dynamic_slice(
        data["n_dens_grid"], (t_start, _i0), (_FAST_T_GRID_SPAN, n_p))
    n_dens_grid_scaled = jax.lax.dynamic_slice(
        data["n_dens_grid_scaled"], (t_start, _i0), (_FAST_T_GRID_SPAN, n_p))

    h_minus_cross = data["h_minus_cross"]
    if data["add_H_minus"]:
        h_minus_cross = jax.lax.dynamic_slice(
            h_minus_cross, (t_start, _i0), (_FAST_T_GRID_SPAN, n_lambda))

    return (T_grid, abundances, all_absorption_data, all_coll_data,
            n_dens_grid, n_dens_grid_scaled, h_minus_cross)


def _jax_compute_base_absorption_coeff(abundances, all_absorption_data,
                                       opacity_species_indices,
                                       all_coll_data, coll_indices,
                                       n_coll_pairs, lambda_grid,
                                       n_dens_grid_scaled):
    """Gas + collisional absorption on the static (T, P, lambda) grid."""
    n_lambda = lambda_grid.shape[0]
    absorption_coeff = jnp.zeros(
        (n_dens_grid_scaled.shape[0], n_dens_grid_scaled.shape[1], n_lambda),
        dtype=_FM_DTYPE)

    if all_absorption_data.shape[0] > 0:
        opacity_abund = abundances[opacity_species_indices]
        # Contract directly over species to avoid materializing the full
        # (n_species, N_T, N_P, N_lambda) product tensor.
        absorption_coeff = absorption_coeff + jnp.einsum(
            "sij,sijl->ijl", opacity_abund, all_absorption_data)

    if n_coll_pairs > 0:
        idx1 = coll_indices[:, 0]
        idx2 = coll_indices[:, 1]
        pair_density = (
            abundances[idx1] * n_dens_grid_scaled[None, :, :] *
            abundances[idx2] * n_dens_grid_scaled[None, :, :])
        absorption_coeff = absorption_coeff + jnp.einsum(
            "ctl,ctp->tpl", all_coll_data, pair_density)

    return absorption_coeff


def _jax_compute_scattering_base(abundances, all_pol_sqr, n_dens_grid):
    """Return the Rayleigh base term before slope-dependent scaling."""
    sum_pol = jnp.sum(abundances * all_pol_sqr[:, None, None], axis=0)
    return n_dens_grid * sum_pol


def _jax_add_scattering(base_absorption_coeff, scattering_base,
                        lambda_grid, scatt_factor, scatt_slope,
                        ln_scatt_lambda_ratio=None):
    """Add Rayleigh-like scattering to a precomputed gas/collisional baseline."""
    if ln_scatt_lambda_ratio is None:
        ln_scatt_lambda_ratio = jnp.log(_SCATT_REF / lambda_grid)
    scatt_log_scale = (
        scatt_slope * ln_scatt_lambda_ratio[None, None, :]
        - _FM_DTYPE(4.0) * _LN_SCATT_REF
    )
    rayleigh = (scatt_factor * _RAYLEIGH_CONST
                * _POL_SQR_INV_SCALE
                * scattering_base[:, :, None]
                * jnp.exp(scatt_log_scale))
    return base_absorption_coeff + rayleigh


def _jax_add_H_minus(absorption_coeff, abundances, h_minus_cross,
                     P_grid, T_grid, el_index, H_index):
    """Add H- bound-free and free-free absorption to the base coefficient.

    h_minus_cross: (N_T, N_lambda) — precomputed k(T, λ) in m^4/N.
    The full absorption coefficient contribution is:
        k(T, λ) * n_el * n_H * P^2 / (k_B * T)
    where n_el, n_H are mixing ratios (abundances).
    """
    # abundances[el_index]: (N_T, N_P), abundances[H_index]: (N_T, N_P)
    n_el = abundances[el_index]   # (N_T, N_P)
    n_H = abundances[H_index]    # (N_T, N_P)
    # P^2 / (k_B * T) prefactor: (N_T, N_P)
    prefactor = n_el * n_H * P_grid[None, :] ** 2 / (_k_B * T_grid[:, None])
    # h_minus_cross: (N_T, N_lambda), prefactor: (N_T, N_P)
    # Result: (N_T, N_P, N_lambda)
    return absorption_coeff + prefactor[:, :, None] * h_minus_cross[:, None, :]


def _jax_compute_opacity_bases(abundances, data, T_grid, P_grid,
                               all_absorption_data, all_coll_data,
                               n_dens_grid, n_dens_grid_scaled,
                               h_minus_cross=None):
    """Compute the shared gas/collisional and scattering bases."""
    with jax.named_scope("opacity_bases"):
        with jax.named_scope("gas_and_collision"):
            base_absorption_coeff = _jax_compute_base_absorption_coeff(
                abundances, all_absorption_data,
                data["opacity_species_indices"],
                all_coll_data, data["coll_indices"],
                data["n_coll_pairs"], data["lambda_grid"],
                n_dens_grid_scaled)
        if data["add_H_minus"] and h_minus_cross is not None:
            with jax.named_scope("H_minus"):
                base_absorption_coeff = _jax_add_H_minus(
                    base_absorption_coeff, abundances, h_minus_cross,
                    P_grid, T_grid,
                    data["el_index"], data["H_index"])
        with jax.named_scope("scattering_base"):
            scattering_base = _jax_compute_scattering_base(
                abundances, data["all_pol_sqr"], n_dens_grid)
        return base_absorption_coeff, scattering_base


def _jax_compute_opacity_bases_auto(abundances, data, T_grid, P_grid,
                                    use_fast_t, t_start):
    """Use the 8-row fast slab only when the full T/P profile fits inside it."""
    n_t = data["N_T"]
    n_p = data["N_P"]
    n_lambda = data["lambda_grid"].shape[0]
    h_minus_cross = data["h_minus_cross"] if data["add_H_minus"] else None

    def fast_branch(_):
        (T_grid_fast, abundances_fast, all_absorption_fast, all_coll_fast,
         n_dens_fast, n_dens_scaled_fast, h_minus_fast) = (
            _jax_slice_temperature_views(abundances, data, t_start))
        base_fast, scatter_fast = _jax_compute_opacity_bases(
            abundances_fast, data, T_grid_fast, P_grid,
            all_absorption_fast, all_coll_fast,
            n_dens_fast, n_dens_scaled_fast, h_minus_fast)

        base_full = jnp.zeros((n_t, n_p, n_lambda), dtype=_FM_DTYPE)
        scatter_full = jnp.zeros((n_t, n_p), dtype=_FM_DTYPE)
        _i0 = jnp.int32(0)
        base_full = jax.lax.dynamic_update_slice(base_full, base_fast, (t_start, _i0, _i0))
        scatter_full = jax.lax.dynamic_update_slice(scatter_full, scatter_fast, (t_start, _i0))
        return base_full, scatter_full

    def full_branch(_):
        return _jax_compute_opacity_bases(
            abundances, data, T_grid, P_grid,
            data["all_absorption_data"], data["all_coll_data"],
            data["n_dens_grid"], data["n_dens_grid_scaled"],
            h_minus_cross)

    return jax.lax.cond(use_fast_t, fast_branch, full_branch, operand=None)


def _jax_integrate_transit_depths(absorption_coeff, T_grid, P_grid,
                                  T_profile, radii, dr, Rs,
                                  log_cloudtop_P, is_isothermal, data,
                                  dl=None):
    """Interpolate, apply cloud opacity, and integrate to unbinned depths."""
    with jax.named_scope("integrate_transit_depths"):
        P_profile = data["P_profile"]
        with jax.named_scope("interpolate_to_layers"):
            if is_isothermal:
                absorption_coeff_atm = _jax_interpolate_to_profile_isothermal(
                    absorption_coeff, T_grid, P_grid, T_profile, P_profile,
                    data["profile_p_lo"], data["profile_p_hi"], data["profile_p_frac"],
                    data["ln_P_profile"], data["ln_P_grid"], data["ln_kBT_grid"])
            else:
                absorption_coeff_atm = _jax_interpolate_to_profile(
                    absorption_coeff, T_grid, P_grid, T_profile, P_profile,
                    data["profile_p_lo"], data["profile_p_hi"], data["profile_p_frac"],
                    data["ln_P_profile"], data["ln_P_grid"], data["ln_kBT_grid"])
        '''
        with jax.named_scope("cloud_opacity"):
            log10_P = data["log10_P_profile"]
            cloud_opacity = 100.0 * jax.nn.sigmoid(500.0 * (log10_P - log_cloudtop_P))

        with jax.named_scope("tau_los"):
            if dl is None:
                dl = _jax_get_dl(radii)
            tau_los = _jax_get_line_of_sight_tau_from_dl(absorption_coeff_atm, dl)
            cloud_tau_los = jnp.dot(0.5 * (cloud_opacity[:-1] + cloud_opacity[1:]), dl)
            tau_los = tau_los + cloud_tau_los[None, :]
        with jax.named_scope("transit_depth_integral"):
            absorption_fraction = -jnp.expm1(-tau_los)
            return ((radii[-1] / Rs) ** 2 +
                    2.0 / Rs ** 2 * jnp.dot(absorption_fraction, radii[1:] * dr))
        '''
        with jax.named_scope("hard_cloud_cutoff"):
            log10_P = data["log10_P_profile"]                  # (N_layers,)
            active = log10_P < log_cloudtop_P                 # True above cloudtop

            # Zero opacity below cloudtop.
            absorption_coeff_atm = jnp.where(
                active[:, None],
                absorption_coeff_atm,
                _FM_DTYPE(0.0)
            )

            # Number of active layers. Clip so indexing is always safe.
            n_active = jnp.clip(
                jnp.sum(active).astype(jnp.int32),
                1,
                radii.shape[0]
            )

            # Deepest active radius becomes the new solid-body baseline.
            baseline_r = radii[n_active - 1]

            # Only shells fully above the cloudtop contribute to the annulus.
            shell_mask = jnp.arange(dr.shape[0]) < (n_active - 1)
            shell_weights = jnp.where(
                shell_mask,
                radii[1:] * dr,
                _FM_DTYPE(0.0)
            )

        with jax.named_scope("tau_los"):
            if dl is None:
                dl = _jax_get_dl(radii)

            # No grey sigmoid cloud opacity term anymore.
            tau_los = _jax_get_line_of_sight_tau_from_dl(absorption_coeff_atm, dl)

        with jax.named_scope("transit_depth_integral"):
            absorption_fraction = -jnp.expm1(-tau_los)
            return ((baseline_r / Rs) ** 2 +
                    2.0 / Rs ** 2 * jnp.dot(absorption_fraction, shell_weights))
        

def _jax_compute_unbinned_transit_depths_from_bases(
        base_absorption_coeff, scattering_base,
        lambda_grid, scatt_factor, scatt_slope,
        T_grid, P_grid, T_profile,
        radii, dr, Rs, log_cloudtop_P, is_isothermal, data, dl=None):
    """Compute unbinned depths from precomputed gas/collisional bases."""
    absorption_coeff = _jax_add_scattering(
        base_absorption_coeff, scattering_base,
        lambda_grid, scatt_factor, scatt_slope,
        data["ln_scatt_lambda_ratio"])
    return _jax_integrate_transit_depths(
        absorption_coeff, T_grid, P_grid, T_profile,
        radii, dr, Rs, log_cloudtop_P, is_isothermal, data, dl=dl)


def _jax_compute_patchy_unbinned_depths_from_bases(
        base_absorption_coeff, scattering_base,
        lambda_grid, cloudy_scatt_factor, cloudy_scatt_slope,
        T_grid, P_grid, T_profile,
        radii, dr, Rs, log_cloudtop_P, is_isothermal, data):
    """Compute clear/cloudy unbinned depths from shared opacity bases."""
    dl = _jax_get_dl(radii)

    cloudy_absorption = _jax_add_scattering(
        base_absorption_coeff, scattering_base,
        lambda_grid, cloudy_scatt_factor, cloudy_scatt_slope,
        data["ln_scatt_lambda_ratio"])
    depths_cloudy = _jax_integrate_transit_depths(
        cloudy_absorption, T_grid, P_grid, T_profile,
        radii, dr, Rs, log_cloudtop_P, is_isothermal, data, dl=dl)

    clear_absorption = _jax_add_scattering(
        base_absorption_coeff, scattering_base,
        lambda_grid, _FM_DTYPE(1.0), _FM_DTYPE(4.0),
        data["ln_scatt_lambda_ratio"])
    depths_clear = _jax_integrate_transit_depths(
        clear_absorption, T_grid, P_grid, T_profile,
        radii, dr, Rs, _FM_DTYPE(jnp.inf), is_isothermal, data, dl=dl)
    return depths_cloudy, depths_clear


def _jax_compute_stellar_spectra(params_dict, data):
    """Return (composite stellar spectrum, transit correction factors).

    When ``dynamic_stellar`` is False these are the static arrays baked in by
    ``prepare_jax_data``; otherwise the spot-weighted spectrum is rebuilt from
    the current T_star / T_spot / spot_cov_frac.
    """
    if not data["dynamic_stellar"]:
        return data["stellar_spectrum"], data["correction_factors"]

    T_star = (_FM_DTYPE(params_dict["T_star"])
              if data.get("fit_T_star", False)
              else _FM_DTYPE(data["default_T_star"]))
    T_spot = _FM_DTYPE(params_dict.get("T_spot", T_star))
    spot_cov_frac = _FM_DTYPE(params_dict.get("spot_cov_frac", 0.0))
    if data.get("stellar_blackbody", False):
        # Static flag: always use blackbody
        spot_spectrum = _jax_blackbody_stellar_spectrum(
            T_spot, data["lambda_grid"])
        unspotted = _jax_blackbody_stellar_spectrum(
            T_star, data["lambda_grid"])
    elif data.get("fit_T_star", False):
        # T_star is traced — must handle out-of-grid via jnp.where
        use_bb = ((T_star < data["stellar_spectra_temps"][0])
                  | (T_star > data["stellar_spectra_temps"][-1]))
        stellar_temps = jnp.stack([T_spot, T_star])
        grid_spot, grid_unspotted = jax_interp1d(
            stellar_temps, data["stellar_spectra_temps"],
            data["stellar_spectra_grid"])
        bb_spot = _jax_blackbody_stellar_spectrum(
            T_spot, data["lambda_grid"])
        bb_unspotted = _jax_blackbody_stellar_spectrum(
            T_star, data["lambda_grid"])
        spot_spectrum = jnp.where(use_bb, bb_spot, grid_spot)
        unspotted = jnp.where(use_bb, bb_unspotted, grid_unspotted)
    else:
        # Use jnp.where so this is safe under JAX tracing
        use_bb = ((T_star < data["stellar_spectra_temps"][0])
                  | (T_star > data["stellar_spectra_temps"][-1]))
        bb_spot = _jax_blackbody_stellar_spectrum(
            T_spot, data["lambda_grid"])
        bb_unspotted = _jax_blackbody_stellar_spectrum(
            T_star, data["lambda_grid"])
        grid_spot = jax_interp1d(
            T_spot, data["stellar_spectra_temps"],
            data["stellar_spectra_grid"])[0]
        spot_spectrum = jnp.where(use_bb, bb_spot, grid_spot)
        unspotted = data["unspotted_spectrum"]
    stellar_spectrum = (spot_cov_frac * spot_spectrum +
                        (1 - spot_cov_frac) * unspotted)
    correction_factors = unspotted / jnp.maximum(stellar_spectrum, _EPS)
    return stellar_spectrum, correction_factors


def _jax_apply_stellar_and_bin(transit_depths, params_dict, data):
    """Apply stellar correction, binning, and transit offsets."""
    with jax.named_scope("stellar_and_binning"):
        with jax.named_scope("stellar_correction"):
            stellar_spectrum, correction_factors = _jax_compute_stellar_spectra(
                params_dict, data)

        with jax.named_scope("binning"):
            corrected = transit_depths * correction_factors
            corrected_expanded = corrected[data["expanded_lambda_idx"]]
            if data["dynamic_stellar"]:
                current_weights = stellar_spectrum[data["expanded_lambda_idx"]]
                weight_sums = jax.ops.segment_sum(
                    current_weights, data["bin_ids"],
                    num_segments=data["n_bins"])
                normalized_weights = current_weights / jnp.maximum(
                    weight_sums[data["bin_ids"]], _EPS)
                weighted = corrected_expanded * normalized_weights
            else:
                weighted = corrected_expanded * data["bin_weights_1d"]
            binned = jax.ops.segment_sum(weighted, data["bin_ids"],
                                         num_segments=data["n_bins"])

        with jax.named_scope("offsets"):
            offset_names = data["offset_names"]
            if offset_names:
                offset_vals = jnp.array(
                    [_FM_DTYPE(params_dict.get(name, 0.0)) for name in offset_names])
                binned = binned + jnp.dot(offset_vals, data["offset_masks"])
        return binned


# ---------------------------------------------------------------------------
# Full forward model:  params → binned transit depths
# ---------------------------------------------------------------------------

def jax_compute_transit_depths(params_dict, data):
    """Compute binned transit depths.  Pure JAX, fully differentiable."""
    with jax.named_scope("forward_model"):
        scatt_factor = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_scatt_factor"])
        scatt_slope  = _FM_DTYPE(params_dict["scatt_slope"])
        Rs, T_grid, P_grid, P_profile, T_profile, abundances, radii, dr, is_unbound = (
            _jax_build_atmosphere_state(params_dict, data))
        def unbound_branch(_):
            return jnp.full((data["n_bins"],), jnp.nan, dtype=_FM_DTYPE)

        def bound_branch(_):
            is_isothermal = data.get("profile_type", "isothermal") == "isothermal"
            use_fast_t, t_start = _jax_get_temperature_slab(
                T_profile, T_grid, data.get("disable_fast_t", False))

            with jax.named_scope("temperature_slab_selection"):
                base_absorption_coeff, scattering_base = _jax_compute_opacity_bases_auto(
                    abundances, data, T_grid, P_grid, use_fast_t, t_start)
            transit_depths = _jax_compute_unbinned_transit_depths_from_bases(
                base_absorption_coeff, scattering_base,
                data["lambda_grid"], scatt_factor, scatt_slope,
                T_grid, P_grid, T_profile,
                radii, dr, Rs, _FM_DTYPE(params_dict["log_cloudtop_P"]),
                is_isothermal, data)

            return _jax_apply_stellar_and_bin(transit_depths, params_dict, data)

        return jax.lax.cond(is_unbound, unbound_branch, bound_branch, operand=None)


# ---------------------------------------------------------------------------
# Patchy cloud wrapper
# ---------------------------------------------------------------------------

def jax_compute_transit_depths_patchy(params_dict, data):
    """Compute transit depths with patchy cloud model.

    Runs the forward model twice (clear + cloudy) and blends by
    cloud_cov_frac, matching TransitDepthCalculator.compute_depths_patchy().
    """
    with jax.named_scope("patchy_forward_model"):
        cloud_cov_frac = _FM_DTYPE(params_dict.get("cloud_cov_frac", 1.0))

        Rs, T_grid, P_grid, P_profile, T_profile, abundances, radii, dr, is_unbound = (
            _jax_build_atmosphere_state(params_dict, data))
        def unbound_branch(_):
            return jnp.full((data["n_bins"],), jnp.nan, dtype=_FM_DTYPE)

        def bound_branch(_):
            is_isothermal = data.get("profile_type", "isothermal") == "isothermal"
            use_fast_t, t_start = _jax_get_temperature_slab(
                T_profile, T_grid, data.get("disable_fast_t", False))
            cloudy_scatt_factor = _FM_DTYPE(10.0) ** _FM_DTYPE(params_dict["log_scatt_factor"])
            cloudy_scatt_slope = _FM_DTYPE(params_dict["scatt_slope"])

            with jax.named_scope("temperature_slab_selection"):
                base_absorption_coeff, scattering_base = _jax_compute_opacity_bases_auto(
                    abundances, data, T_grid, P_grid, use_fast_t, t_start)
            depths_cloudy, depths_clear = _jax_compute_patchy_unbinned_depths_from_bases(
                base_absorption_coeff, scattering_base,
                data["lambda_grid"], cloudy_scatt_factor, cloudy_scatt_slope,
                T_grid, P_grid, T_profile,
                radii, dr, Rs, _FM_DTYPE(params_dict["log_cloudtop_P"]),
                is_isothermal, data)

            with jax.named_scope("blend_patchy"):
                blended_depths = (
                    cloud_cov_frac * depths_cloudy + (1 - cloud_cov_frac) * depths_clear)
            return _jax_apply_stellar_and_bin(blended_depths, params_dict, data)

        return jax.lax.cond(is_unbound, unbound_branch, bound_branch, operand=None)


# ---------------------------------------------------------------------------
# 1.5D limb-asymmetric model
# ---------------------------------------------------------------------------

def jax_compute_transit_depths_limb_asym(params_dict, data):
    """1.5D limb-asymmetric transit model.

    Morning limb uses the base T/P profile with clouds at log_cloudtop_P.
    Evening limb uses the base profile shifted by +delta_T with its own
    cloud deck and its own Rayleigh-like scattering parameters. Chemistry,
    Rp, Mp, Rs are shared.

    Each sector is evaluated as a *full* transit depth (as if the entire
    planet had that limb's properties), and the two full depths are
    linearly combined as

        depths = f_limb * morning + (1 - f_limb) * evening

    This is the standard 1.5D formulation (MacDonald & Lewis 2022 Aurora,
    POSEIDON): splitting the terminator into angular fractions f_limb and
    (1 - f_limb) means the blocked-area integral around the annulus is a
    direct weighted average of the two full sector depths. When all
    parameters are identical (delta_T = 0, same clouds/scatt, f_limb any),
    this collapses to the 1D model exactly.

    The opacity bases (gas absorption, CIA, scattering) are computed once
    on the master (T, P, lambda) grid when chemistry is identical, then
    the integration to unbinned transit depths is performed twice with
    separate T/P profiles and hydrostatic structures.
    """
    with jax.named_scope("limb_asym_forward_model"):
        f_limb  = _FM_DTYPE(params_dict.get("f_limb", 0.5))
        delta_T = _FM_DTYPE(params_dict.get("delta_T", 0.0))

        Rp = _FM_DTYPE(params_dict["Rp"])
        Rs = _FM_DTYPE(params_dict["Rs"])
        Mp = _FM_DTYPE(params_dict["Mp"])
        T_grid    = data["T_grid"]
        P_grid    = data["P_grid"]
        P_profile = data["P_profile"]
        ln_P_profile = data["ln_P_profile"]

        # --- temperature profiles for each sector ---
        T_profile_morning = _jax_get_temperature_profile(params_dict, data)
        T_profile_evening = T_profile_morning + delta_T

        # --- chemistry on master grid ---
        # Chemistry is shared across limbs except when quenching is enabled
        # and T_quench is derived from each limb's own T/P profile. In that
        # case the quenched abundances become profile-dependent and must be
        # recomputed separately for morning and evening.
        profile_dependent_quench = (
            data.get("quench_enabled", False)
            and not data.get("fit_T_quench", False)
        )
        abundances_morning = _jax_get_master_grid_abundances(
            params_dict, data, T_profile_morning)
        if profile_dependent_quench:
            abundances_evening = _jax_get_master_grid_abundances(
                params_dict, data, T_profile_evening)
        else:
            abundances_evening = abundances_morning

        # --- separate hydrostatic solves per sector ---
        # Each limb gets its own pressure-radius mapping (different T →
        # different scale height → different atmospheric extent), following
        # MacDonald & Lewis (2022) / Espinoza & Jones (2021).  Both share
        # Rp at the reference pressure.  The linear blend uses a shared
        # baseline (radii_m[-1]) to avoid double-counting the solid body.
        _, mu_morning = _jax_interp_abundances_to_profile(
            abundances_morning, data["all_masses"], T_grid, P_grid,
            T_profile_morning, P_profile,
            data["profile_p_lo"], data["profile_p_hi"], data["profile_p_frac"])
        T_star = params_dict.get("T_star")
        morning_unbound = _jax_atmosphere_is_unbound(
            T_profile_morning, mu_morning, Mp, Rp, Rs,
            None if T_star is None else _FM_DTYPE(T_star))
        radii_m, dr_m = jax.lax.cond(
            morning_unbound,
            lambda _: (jnp.full_like(P_profile, jnp.nan),
                       jnp.full((P_profile.shape[0] - 1,), jnp.nan, dtype=_FM_DTYPE)),
            lambda _: _jax_hydrostatic_solve(
                ln_P_profile, T_profile_morning, mu_morning, Mp, Rp,
                data["ln_P_below"], data["ln_P_above"]),
            operand=None)

        _, mu_evening = _jax_interp_abundances_to_profile(
            abundances_evening, data["all_masses"], T_grid, P_grid,
            T_profile_evening, P_profile,
            data["profile_p_lo"], data["profile_p_hi"], data["profile_p_frac"])
        evening_unbound = _jax_atmosphere_is_unbound(
            T_profile_evening, mu_evening, Mp, Rp, Rs,
            None if T_star is None else _FM_DTYPE(T_star))
        radii_e, dr_e = jax.lax.cond(
            evening_unbound,
            lambda _: (jnp.full_like(P_profile, jnp.nan),
                       jnp.full((P_profile.shape[0] - 1,), jnp.nan, dtype=_FM_DTYPE)),
            lambda _: _jax_hydrostatic_solve(
                ln_P_profile, T_profile_evening, mu_evening, Mp, Rp,
                data["ln_P_below"], data["ln_P_above"]),
            operand=None)
        is_unbound = morning_unbound | evening_unbound

        # --- temperature slab covering BOTH profiles ---
        if data.get("disable_fast_t", False):
            t_start = jnp.int32(0)
            use_fast_t = False
        else:
            t_min = jnp.minimum(jnp.min(T_profile_morning),
                                jnp.min(T_profile_evening))
            t_max = jnp.maximum(jnp.max(T_profile_morning),
                                jnp.max(T_profile_evening))
            t_start = jnp.maximum(
                jnp.searchsorted(T_grid, t_min, side="left") - 1, 0)
            t_end = jnp.minimum(
                jnp.searchsorted(T_grid, t_max, side="left") + 1,
                T_grid.shape[0])
            t_start = jnp.minimum(
                t_start, T_grid.shape[0] - _FAST_T_GRID_SPAN).astype(jnp.int32)
            use_fast_t = (t_end - t_start) <= _FAST_T_GRID_SPAN

        is_isothermal = data.get("profile_type", "isothermal") == "isothermal"

        def unbound_branch(_):
            return jnp.full((data["n_bins"],), jnp.nan, dtype=_FM_DTYPE)

        def bound_branch(_):
            with jax.named_scope("temperature_slab_selection"):
                (base_absorption_coeff_m,
                 scattering_base_m) = _jax_compute_opacity_bases_auto(
                    abundances_morning, data, T_grid, P_grid, use_fast_t, t_start)
                if profile_dependent_quench:
                    (base_absorption_coeff_e,
                     scattering_base_e) = _jax_compute_opacity_bases_auto(
                        abundances_evening, data, T_grid, P_grid, use_fast_t, t_start)
                else:
                    base_absorption_coeff_e = base_absorption_coeff_m
                    scattering_base_e = scattering_base_m

            morning_scatt_factor = _FM_DTYPE(10.0) ** _FM_DTYPE(
                params_dict["log_scatt_factor"])
            morning_scatt_slope = _FM_DTYPE(params_dict["scatt_slope"])
            morning_absorption = _jax_add_scattering(
                base_absorption_coeff_m, scattering_base_m,
                data["lambda_grid"], morning_scatt_factor, morning_scatt_slope,
                data["ln_scatt_lambda_ratio"])
            depths_morning = _jax_integrate_transit_depths(
                morning_absorption, T_grid, P_grid,
                T_profile_morning, radii_m, dr_m, Rs,
                _FM_DTYPE(params_dict["log_cloudtop_P"]),
                is_isothermal, data)

            evening_scatt_factor = _FM_DTYPE(10.0) ** _FM_DTYPE(
                params_dict.get("log_scatt_factor_evening", 0.0))
            evening_scatt_slope = _FM_DTYPE(
                params_dict.get("scatt_slope_evening", 4.0))
            evening_absorption = _jax_add_scattering(
                base_absorption_coeff_e, scattering_base_e,
                data["lambda_grid"], evening_scatt_factor, evening_scatt_slope,
                data["ln_scatt_lambda_ratio"])
            log_ctp_eve = _FM_DTYPE(params_dict.get("log_cloudtop_P_evening", jnp.inf))
            depths_evening = _jax_integrate_transit_depths(
                evening_absorption, T_grid, P_grid,
                T_profile_evening, radii_e, dr_e, Rs,
                log_ctp_eve,
                is_isothermal, data)

            with jax.named_scope("blend_limb_asym"):
                # 1.5D: each sector's depth is already a complete transit
                # depth (solid body + annular). Angular-weighted average of
                # the two full depths is the physically correct blend.
                blended = f_limb * depths_morning + (1 - f_limb) * depths_evening

            return _jax_apply_stellar_and_bin(blended, params_dict, data)

        return jax.lax.cond(is_unbound, unbound_branch, bound_branch, operand=None)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def jax_log_likelihood(params_arr, param_names, all_param_defaults, data,
                       measured_depths, measured_errors):
    """Scalar log-likelihood from a flat parameter vector.

    Forward model runs in float32 for GPU speed; likelihood sum is cast
    to float64 for HMC numerical stability.
    """
    params_dict = dict(all_param_defaults)
    for i, name in enumerate(param_names):
        params_dict[name] = params_arr[i]

    if data.get("limb_asym", False):
        binned = jax_compute_transit_depths_limb_asym(params_dict, data)
    elif data.get("patchy", False):
        binned = jax_compute_transit_depths_patchy(params_dict, data)
    else:
        binned = jax_compute_transit_depths(params_dict, data)

    # Compute residuals in float32 (matching binned), then cast to float64
    # for the reduction to avoid accumulation errors over 4440 bins.
    if "error_additive" in param_names:
        err_add = _FM_DTYPE(params_dict["error_additive"])
        scaled_err = jnp.sqrt(measured_errors.astype(_FM_DTYPE)**2 + err_add**2)
    else:
        err_mult = _FM_DTYPE(params_dict["error_multiple"])
        scaled_err = err_mult * measured_errors.astype(_FM_DTYPE)
    resid = binned - measured_depths.astype(_FM_DTYPE)
    per_point = resid ** 2 / scaled_err ** 2 + jnp.log(
        _FM_DTYPE(2 * jnp.pi) * scaled_err ** 2)
    return -0.5 * jnp.sum(per_point)
