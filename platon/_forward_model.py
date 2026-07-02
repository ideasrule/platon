"""JAX forward models for transit and eclipse depths.

Everything here runs in FP32 (jax_enable_x64 is False).  Care is taken to
avoid FP32 overflow/underflow:

- Cross sections are handled in log space, with a floor of ln(1e-99); linear
  cross sections as small as 1e-99 are NOT representable in FP32.
- Collision-induced absorption data (~1e-63..1e-54) is stored pre-multiplied
  by CIA_DATA_SCALE=1e56, and the two number densities it multiplies are each
  scaled by N_SCALE=1e-28, keeping every intermediate in FP32 range.
- Polarizabilities (~1e-30) are stored scaled by POL_SCALE=1e30 and the
  Rayleigh/haze scattering power law is evaluated with wavelengths in microns,
  so that sum_pol_sqr and lambda**slope stay in range for slopes up to 15.
"""
import math
from typing import NamedTuple, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .constants import k_B, AMU, G, h, c, M_sun, Teff_sun
from ._interpolator_3D import regular_grid_interp, interp1d

N_SCALE = 1e-28          # scaling of number densities in the CIA term
CIA_DATA_SCALE = 1e56    # compensates N_SCALE**2 in the stored CIA data
POL_SCALE = 1e30         # polarizabilities are stored multiplied by this
# 128/3 * pi^5 * POL_SCALE^-2 * (1e6 m/um)^-24-compensation folded together:
# coeff = factor * RAYLEIGH_PREF * n * sum((pol*POL_SCALE)^2 * abund)
#         * ref_um^(slope-4) / lambda_um^slope
RAYLEIGH_PREF = 128.0 / 3 * math.pi ** 5 * 1e-36
LOG_MIN_ABUND = -99.0
TWO_H_C_SQR = 2 * h * c ** 2
HC_OVER_KB = h * c / k_B

# Indices into the packed scalar-parameter vector
(SC_RS, SC_MP, SC_RP, SC_LOGZ, SC_CO, SC_LOG_CH4, SC_SCAT_FACTOR,
 SC_SCAT_SLOPE, SC_SCAT_REF_UM, SC_CLOUDTOP, SC_T_QUENCH, SC_P_QUENCH,
 SC_LOG10_P_QUENCH, SC_T_STAR, SC_T_SPOT, SC_SPOT_FRAC, SC_FSH, SC_NUM_DEN,
 SC_LN_MIN_XSEC, SC_LOG_MIN_ABUND, SC_REF_PRESSURE, SC_T_STAR_HYDRO,
 SC_MIE_REF_P, SC_SURFACE_P, SC_A_OVER_RS, SC_SURFACE_TEMP, SC_REDIST,
 SC_N_SCALARS) = range(28)

# Indices into the packed int vector
IX_T0, IX_FLOOR, IX_BOT, IX_N_INTS = range(4)


class DeviceData(NamedTuple):
    """All device-resident model data.  Fields are jnp arrays (or None)."""
    lambda_grid: Any        # (L,)
    lambda_um: Any          # (L,)
    T_grid: Any             # (NT,)
    P_grid: Any             # (NP,)
    ln_P_grid: Any          # (NP,)
    log10_P_grid: Any       # (NP,)
    abs_stack: Any          # (S, NT, NP, L) absorption coefficient grids
    opac_master_idx: Any    # (S,) int32: index into master species list
    masses: Any             # (M,) AMU
    pol_sqr: Any            # (M,) (polarizability * POL_SCALE)**2
    log_abund_grid: Any     # (NZ, NC, S_eq, NT, NP) log10 abundances
    logZ_grid: Any          # (NZ,)
    CO_grid: Any            # (NC,)
    eq_master_idx: Any      # (S_eq,) int32
    cia_stack: Any          # (K, NT, L), pre-multiplied by CIA_DATA_SCALE
    cia_idx1: Any           # (K,) int32
    cia_idx2: Any           # (K,) int32
    hminus_k_over_kB: Any   # (NT, L): H- k(T, lambda) / k_B
    stellar_temps: Any      # (NS,)
    stellar_spectra: Any    # (NS, L)
    orig_lambda_grid: Any   # (L0,) full-resolution grid
    orig_stellar_spectra: Any  # (NS, L0)
    exp3_x: Any             # (NE,) tau values for the E3 lookup table
    exp3_y: Any             # (NE,)
    bterm_x: Any            # (NB,) tau values for the bottom-boundary term
    bterm_y: Any            # (NB,) tau^2 E1(tau) - tau e^-tau + e^-tau
    bin_mat_transit: Any    # (B, L) or None
    bin_mat_eclipse: Any    # (B, L) or None


class ForwardConfig(NamedTuple):
    """Static (hashable) configuration; changing any field recompiles."""
    n_master: int
    n_t_rows: int
    abund_mode: str          # 'eq', 'vmr', or 'custom'
    gas_master_idx: tuple    # master indices of fit gases ('vmr' mode)
    ch4_idx: int
    el_idx: int
    h_idx: int
    add_gas: bool
    add_hminus: bool
    add_scattering: bool
    add_collisional: bool
    use_mie: bool
    has_t_star: bool
    blackbody: bool
    has_bins: bool
    has_surface: bool = False
    surface_temp_given: bool = False


class ForwardInputs(NamedTuple):
    scalars: Any            # (SC_N_SCALARS,) float32
    ints: Any               # (IX_N_INTS,) int32
    T_profile: Any          # (N,)
    P_profile: Any          # (N,)
    shell_mask: Any         # (N-1,) 1.0 where the shell is above cloud/surface
    opac_mask: Any          # (S,)
    vmrs: Any = None        # (n_gases,) for 'vmr' mode
    custom_log_abund: Any = None  # (M, NT, NP) for 'custom' mode
    eff_xsec: Any = None    # (L,) Mie effective cross sections
    rh_binned: Any = None   # (L,) surface hemispheric reflectance
    rh_orig: Any = None     # (L0,)
    crust_flux: Any = None  # (NC2,)
    crust_T: Any = None     # (NC2,)


class AtmosphereOutputs(NamedTuple):
    radii: Any
    dr: Any
    mu_profile: Any
    atm_abund: Any          # (N, M)
    absorption_coeff_atm: Any  # (N, L)
    unbound: Any            # bool scalar


class TransitOutputs(NamedTuple):
    binned_depths: Any
    depths: Any             # (L,) uncorrected depths
    stellar_spectrum: Any
    correction_factors: Any
    tau_los: Any            # (L, N-1)
    absorption_fraction: Any
    atm: AtmosphereOutputs


class EclipseOutputs(NamedTuple):
    binned_depths: Any
    depths: Any             # (L,) unbinned eclipse depths
    fluxes: Any             # (L,) planet spectrum
    stellar_spectrum: Any
    taus: Any               # (L, N-1)
    integrand: Any          # (L, N-1)
    photosphere_radii: Any
    surface_temp: Any
    irrad: Any
    atm: AtmosphereOutputs


def _build_log_abundances(cfg, data, sc, inp):
    """Returns (M, NT, NP) log10 abundances on the full T/P grid."""
    n_master = cfg.n_master
    NT, NP = data.T_grid.shape[0], data.P_grid.shape[0]

    if cfg.abund_mode == "eq":
        la_eq = regular_grid_interp(
            data.logZ_grid, data.CO_grid, data.log_abund_grid,
            sc[SC_LOGZ], sc[SC_CO])                     # (S_eq, NT, NP)
        la = jnp.full((n_master, NT, NP), LOG_MIN_ABUND, dtype=jnp.float32)
        la = la.at[data.eq_master_idx].set(la_eq)
        if cfg.ch4_idx >= 0:
            la = la.at[cfg.ch4_idx].add(sc[SC_LOG_CH4])
    elif cfg.abund_mode == "vmr":
        la = jnp.full((n_master, NT, NP), LOG_MIN_ABUND, dtype=jnp.float32)
        idx = jnp.asarray(cfg.gas_master_idx, dtype=jnp.int32)
        vals = jnp.broadcast_to(
            jnp.log10(inp.vmrs)[:, None, None], (len(cfg.gas_master_idx), NT, NP))
        la = la.at[idx].set(vals)
    else:
        la = inp.custom_log_abund

    la = jnp.maximum(la, sc[SC_LOG_MIN_ABUND])

    # Quenching: below P_quench, hold abundances at their (T_quench, P_quench)
    # values
    la_tpm = jnp.transpose(la, (1, 2, 0))               # (NT, NP, M)
    quench_la = regular_grid_interp(
        data.T_grid, data.log10_P_grid, la_tpm,
        sc[SC_T_QUENCH], sc[SC_LOG10_P_QUENCH])         # (M,)
    quench_mask = data.P_grid <= sc[SC_P_QUENCH]        # (NP,)
    la = jnp.where(quench_mask[None, None, :], quench_la[:, None, None], la)
    return la


def _hydrostatic(sc, P_profile, T_profile, mu_profile):
    """Solve the hydrostatic equation.  Returns radii (descending with
    pressure index), dr = -diff(radii) > 0, and an `unbound` flag."""
    Mp = sc[SC_MP]
    Rp = sc[SC_RP]
    Rs = sc[SC_RS]
    ln_P = jnp.log(P_profile)

    T_mid = 0.5 * (T_profile[1:] + T_profile[:-1])
    mu_mid = 0.5 * (mu_profile[1:] + mu_profile[:-1])
    seg = jnp.diff(ln_P) * k_B * T_mid / (G * Mp * mu_mid * AMU)   # (N-1,)
    C = jnp.concatenate([jnp.zeros(1, dtype=seg.dtype), jnp.cumsum(seg)])

    # Integral value at the reference pressure (piecewise-linear T, mu in lnP)
    ln_ref = jnp.log(sc[SC_REF_PRESSURE])
    fi = jnp.interp(ln_ref, ln_P, jnp.arange(len(ln_P), dtype=jnp.float32))
    k = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, len(ln_P) - 2)
    T_ref = jnp.interp(ln_ref, ln_P, T_profile)
    mu_ref = jnp.interp(ln_ref, ln_P, mu_profile)
    seg_partial = (ln_ref - ln_P[k]) * k_B * 0.5 * (T_profile[k] + T_ref) / \
        (G * Mp * 0.5 * (mu_profile[k] + mu_ref) * AMU)
    C_ref = C[k] + seg_partial

    inv_r = 1.0 / Rp + (C - C_ref)
    radii = 1.0 / inv_r
    # dr computed from the exact segment values to avoid FP32 cancellation
    dr = seg / (inv_r[1:] * inv_r[:-1])

    # Unbound-atmosphere diagnostics (same criteria as the FP64 version)
    R_hill = Rs * (sc[SC_T_STAR_HYDRO] / T_profile[0]) ** 2 * \
        (Mp / (3 * M_sun)) ** (1.0 / 3)
    max_r_estimate = 1.0 / (1.0 / Rp + k_B * jnp.median(T_profile) *
                            jnp.log(P_profile[0] / sc[SC_REF_PRESSURE]) /
                            (G * Mp * jnp.mean(mu_profile) * AMU))
    unbound = (max_r_estimate < 0) | (max_r_estimate > R_hill)
    return radii, dr, unbound


def _opacity(cfg, data, sc, inp, abund, T_profile, P_profile):
    """Assemble absorption coefficients on the (sliced) T/P grid, convert to
    log cross sections, and interpolate onto the T/P profile.  Returns the
    per-layer absorption coefficients (N, L)."""
    NP = data.P_grid.shape[0]
    L = data.lambda_grid.shape[0]
    t0 = inp.ints[IX_T0]
    NTr = cfg.n_t_rows

    T_rows = lax.dynamic_slice_in_dim(data.T_grid, t0, NTr)
    abund_rows = lax.dynamic_slice_in_dim(abund, t0, NTr, axis=1)  # (M,NTr,NP)
    n_rows = data.P_grid[None, :] / (k_B * T_rows[:, None])        # (NTr,NP)

    coeff = jnp.zeros((NTr, NP, L), dtype=jnp.float32)

    if cfg.add_gas:
        stack_rows = lax.dynamic_slice_in_dim(data.abs_stack, t0, NTr, axis=1)
        gas_ab = abund_rows[data.opac_master_idx] * inp.opac_mask[:, None, None]
        coeff += jnp.einsum("stpl,stp->tpl", stack_rows, gas_ab)

    if cfg.add_hminus:
        k_rows = lax.dynamic_slice_in_dim(data.hminus_k_over_kB, t0, NTr)
        w = abund_rows[cfg.el_idx] * abund_rows[cfg.h_idx] * \
            (data.P_grid ** 2)[None, :] / T_rows[:, None]          # (NTr,NP)
        coeff += k_rows[:, None, :] * w[:, :, None]

    if cfg.add_scattering:
        if cfg.use_mie:
            factor, slope, ref_um = 1.0, 4.0, 1.0
        else:
            factor = sc[SC_SCAT_FACTOR]
            slope = sc[SC_SCAT_SLOPE]
            ref_um = sc[SC_SCAT_REF_UM]
        sum_pol = jnp.einsum("mtp,m->tp", abund_rows, data.pol_sqr)
        pow_term = ref_um ** (slope - 4) / data.lambda_um ** slope  # (L,)
        coeff += (factor * RAYLEIGH_PREF) * \
            (n_rows * sum_pol)[:, :, None] * pow_term[None, None, :]

    if cfg.use_mie:
        n_mie = sc[SC_NUM_DEN] * \
            (data.P_grid / sc[SC_MIE_REF_P]) ** (1.0 / sc[SC_FSH])  # (NP,)
        coeff += n_mie[None, :, None] * inp.eff_xsec[None, None, :]

    if cfg.add_collisional:
        cia_rows = lax.dynamic_slice_in_dim(data.cia_stack, t0, NTr, axis=1)
        ns = n_rows * N_SCALE
        w = (abund_rows[data.cia_idx1] * abund_rows[data.cia_idx2]) * \
            (ns * ns)[None, :, :]                                  # (K,NTr,NP)
        coeff += jnp.einsum("ktl,ktp->tpl", cia_rows, w)

    # Cross sections vary less with T and P than absorption coefficients, so
    # interpolate cross sections -- in log space, floored at ln(min_cross_sec)
    ln_xsec = jnp.log(coeff) - jnp.log(n_rows)[:, :, None]
    ln_xsec = jnp.maximum(ln_xsec, sc[SC_LN_MIN_XSEC])

    inv_T_asc = 1.0 / T_rows[::-1]
    ln_xsec_atm = regular_grid_interp(
        inv_T_asc, data.ln_P_grid, ln_xsec[::-1],
        1.0 / T_profile, jnp.log(P_profile))                       # (N, L)
    ln_n_atm = jnp.log(P_profile) - jnp.log(k_B * T_profile)
    return jnp.exp(ln_xsec_atm + ln_n_atm[:, None])


def _compute_atmosphere(cfg, data, inp):
    sc = inp.scalars
    T_profile = inp.T_profile
    P_profile = inp.P_profile

    la = _build_log_abundances(cfg, data, sc, inp)
    abund = 10.0 ** la                                   # (M, NT, NP)

    la_tpm = jnp.transpose(la, (1, 2, 0))                # (NT, NP, M)
    log_ab_atm = regular_grid_interp(
        data.T_grid, data.log10_P_grid, la_tpm,
        T_profile, jnp.log10(P_profile))                 # (N, M)
    atm_abund = 10.0 ** log_ab_atm
    mu_profile = atm_abund @ data.masses                 # (N,)

    radii, dr, unbound = _hydrostatic(sc, P_profile, T_profile, mu_profile)
    coeff_atm = _opacity(cfg, data, sc, inp, abund, T_profile, P_profile)
    return AtmosphereOutputs(radii, dr, mu_profile, atm_abund, coeff_atm,
                             unbound)


def _planck(lambda_grid, T):
    """pi-free Planck spectral radiance B_lambda(T); T may broadcast."""
    return TWO_H_C_SQR / lambda_grid ** 5 / \
        jnp.expm1(HC_OVER_KB / (lambda_grid * T))


def _stellar_spectrum(cfg, data, sc, orig=False):
    """Stellar spectrum and spot correction factors on the wavelength grid."""
    lam = data.orig_lambda_grid if orig else data.lambda_grid
    spectra = data.orig_stellar_spectra if orig else data.stellar_spectra
    L = lam.shape[0]
    if not cfg.has_t_star:
        ones = jnp.ones(L, dtype=jnp.float32)
        return ones, ones

    T_star = sc[SC_T_STAR]
    T_spot = sc[SC_T_SPOT]
    f_spot = sc[SC_SPOT_FRAC]

    in_grid = (T_star >= data.stellar_temps[0]) & \
              (T_star <= data.stellar_temps[-1]) & (not cfg.blackbody)
    unspotted = jnp.where(
        in_grid, interp1d(T_star, data.stellar_temps, spectra),
        math.pi * _planck(lam, T_star))
    spot = jnp.where(
        in_grid, interp1d(T_spot, data.stellar_temps, spectra),
        math.pi * _planck(lam, T_spot))

    spectrum = f_spot * spot + (1 - f_spot) * unspotted
    correction_factors = unspotted / spectrum
    return spectrum, correction_factors


def _get_dl(radii):
    """dl[i, j]: distance travelled by a ray with impact parameter radii[j+1]
    between the shells at radii[i+1] and radii[i]. (radii descending)"""
    # (a - b) * (a + b) instead of a**2 - b**2 avoids FP32 cancellation
    a = radii[:, None]        # r_prime (shell), varying along rows
    b = radii[None, 1:]       # ray impact parameters
    sqr_length = jnp.maximum((a - b) * (a + b), 0.0)
    lengths = 2 * jnp.sqrt(sqr_length)
    return lengths[:-1] - lengths[1:]


def _transit_core(cfg, data, inp):
    sc = inp.scalars
    atm = _compute_atmosphere(cfg, data, inp)
    Rs = sc[SC_RS]

    intermediate_coeff = 0.5 * (atm.absorption_coeff_atm[:-1] +
                                atm.absorption_coeff_atm[1:])  # (N-1, L)
    dl = _get_dl(atm.radii)                                    # (N-1, N-1)
    tau_los = intermediate_coeff.T @ dl                        # (L, N-1)
    absorption_fraction = -jnp.expm1(-tau_los)

    shell_w = inp.shell_mask * atm.radii[1:] * atm.dr
    r_floor = atm.radii[inp.ints[IX_FLOOR]]
    depths = (r_floor / Rs) ** 2 + \
        2.0 / Rs ** 2 * (absorption_fraction @ shell_w)

    stellar, corr = _stellar_spectrum(cfg, data, sc)
    if cfg.has_bins:
        weighted = data.bin_mat_transit @ (depths * corr * stellar)
        norm = data.bin_mat_transit @ stellar
        binned = weighted / norm
    else:
        binned = depths * corr

    return TransitOutputs(binned, depths, stellar, corr, tau_los,
                          absorption_fraction, atm)


def _eclipse_core(cfg, data, inp):
    sc = inp.scalars
    atm = _compute_atmosphere(cfg, data, inp)
    Rs = sc[SC_RS]
    Rp = sc[SC_RP]

    intermediate_coeff = 0.5 * (atm.absorption_coeff_atm[:-1] +
                                atm.absorption_coeff_atm[1:])   # (N-1, L)
    intermediate_T = 0.5 * (inp.T_profile[:-1] + inp.T_profile[1:])
    d_taus = intermediate_coeff.T * (atm.dr * inp.shell_mask)[None, :]
    taus = jnp.cumsum(d_taus, axis=1)                           # (L, N-1)

    planck = _planck(data.lambda_grid[:, None], intermediate_T[None, :])

    exp3 = jnp.interp(taus.ravel(), data.exp3_x, data.exp3_y,
                      left=0.5, right=0.0).reshape(taus.shape)
    exp3_padded = jnp.concatenate(
        [jnp.full((taus.shape[0], 1), 0.5, dtype=taus.dtype), exp3], axis=1)
    integrand = planck * jnp.diff(exp3_padded, axis=1)
    fluxes = -2 * math.pi * jnp.sum(integrand, axis=1)

    max_taus = jnp.max(taus, axis=1)
    # tau^2 E1(tau) - tau e^-tau + e^-tau via a float64-precomputed lookup
    # table.  (jax.scipy.special.exp1's iterative implementation can fail to
    # converge in FP32; the limits are exactly 1 as tau->0 and 0 as tau->inf.)
    bottom_term = jnp.interp(max_taus, data.bterm_x, data.bterm_y,
                             left=1.0, right=0.0)
    planck_bot = jnp.take(planck, inp.ints[IX_BOT], axis=1)

    cloudtop = sc[SC_CLOUDTOP]
    surface_P = sc[SC_SURFACE_P]
    w_cloud = jnp.isfinite(cloudtop) & (cloudtop < surface_P)
    fluxes = fluxes + jnp.where(w_cloud, 1.0, 0.0) * \
        math.pi * planck_bot * bottom_term

    stellar, _ = _stellar_spectrum(cfg, data, sc)

    surface_temp = jnp.float32(0.0)
    irrad = jnp.float32(0.0)
    if cfg.has_surface:
        if cfg.surface_temp_given:
            surface_temp = sc[SC_SURFACE_TEMP]
        else:
            stellar_orig, _ = _stellar_spectrum(cfg, data, sc, orig=True)
            irrad = sc[SC_REDIST] * jnp.trapezoid(
                (1 - inp.rh_orig) * stellar_orig / sc[SC_A_OVER_RS] ** 2,
                data.orig_lambda_grid)
            surface_temp = jnp.interp(irrad, inp.crust_flux, inp.crust_T)
        emitted = (1 - inp.rh_binned) * math.pi * \
            _planck(data.lambda_grid, surface_temp)
        reflected = stellar / sc[SC_A_OVER_RS] ** 2 * inp.rh_binned
        w_surf = surface_P < cloudtop
        fluxes = fluxes + jnp.where(w_surf, 1.0, 0.0) * \
            (emitted + reflected) * bottom_term

    # Photosphere radius: shell where tau is closest to 1 (excluded shells
    # are masked out); planet_radius where the atmosphere is transparent
    abs_ln_tau = jnp.where(inp.shell_mask > 0, jnp.abs(jnp.log(taus)), jnp.inf)
    photosphere_radii = atm.radii[jnp.argmin(abs_ln_tau, axis=1)]
    photosphere_radii = jnp.where(max_taus < 1, Rp, photosphere_radii)

    depths = fluxes / stellar * (photosphere_radii / Rs) ** 2

    if cfg.has_bins:
        photon_w = stellar * data.lambda_grid  # proportional to photon flux
        weighted = data.bin_mat_eclipse @ (depths * photon_w)
        norm = data.bin_mat_eclipse @ photon_w
        binned = weighted / norm
    else:
        binned = depths

    return EclipseOutputs(binned, depths, fluxes, stellar, taus, integrand,
                          photosphere_radii, surface_temp, irrad, atm)


transit_core = jax.jit(_transit_core, static_argnums=0)
eclipse_core = jax.jit(_eclipse_core, static_argnums=0)
