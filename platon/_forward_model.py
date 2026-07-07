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

All per-layer: opacities and abundances are computed on the 1D grid of
atmospheric layers, never on the 2D temperature-pressure grid.
"""
import math
from typing import NamedTuple, Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .constants import k_B, AMU, G, h, c, M_sun
from ._interpolator_3D import regular_grid_interp, interp1d, fractional_index

N_SCALE = 1e-28          # scaling of number densities in the CIA term
CIA_DATA_SCALE = 1e56    # compensates N_SCALE**2 in the stored CIA data
POL_SCALE = 1e30         # polarizabilities are stored multiplied by this
# 128/3 * pi^5 * POL_SCALE^-2 * (1e6 m/um)^-24-compensation folded together:
# coeff = factor * RAYLEIGH_PREF * n * sum((pol*POL_SCALE)^2 * abund)
#         * ref_um^(slope-4) / lambda_um^slope
RAYLEIGH_PREF = 128.0 / 3 * math.pi ** 5 * 1e-36
LOG_MIN_ABUND = -99.0
LN_MIN_XSEC = math.log(1e-99)   # floor of the stored log cross sections
TWO_H_C_SQR = 2 * h * c ** 2
HC_OVER_KB = h * c / k_B

# Indices into the packed scalar-parameter vector
(SC_RS, SC_MP, SC_RP, SC_LOGZ, SC_CO, SC_LOG_CH4, SC_SCAT_FACTOR,
 SC_SCAT_SLOPE, SC_SCAT_REF_UM, SC_CLOUDTOP, SC_P_QUENCH,
 SC_LOG10_P_QUENCH, SC_T_STAR, SC_T_SPOT, SC_SPOT_FRAC, SC_FSH, SC_NUM_DEN,
 SC_LN_MIN_XSEC, SC_LOG_MIN_ABUND, SC_REF_PRESSURE, SC_T_STAR_HYDRO,
 SC_MIE_REF_P, SC_SURFACE_P, SC_A_OVER_RS, SC_SURFACE_TEMP, SC_REDIST,
 SC_N_SCALARS) = range(27)

# Indices into the packed int vector
IX_FLOOR, IX_N_INTS = range(2)


class DeviceData(NamedTuple):
    """All device-resident model data.  Fields are jnp arrays (or None)."""
    lambda_grid: Any        # (L,)
    lambda_um: Any          # (L,)
    T_grid: Any             # (NT,)
    P_grid: Any             # (NP,)
    ln_P_grid: Any          # (NP,)
    log10_P_grid: Any       # (NP,)
    inv_T_grid: Any         # (NT,) 1 / T_grid
    ln_xsec_stack: Any      # (S, NT, NP, L) ln cross sections per species
    opac_master_idx: Any    # (S,) int32: index into master species list
    masses: Any             # (M,) AMU
    pol_sqr: Any            # (M,) (polarizability * POL_SCALE)**2
    log_abund_grid: Any     # (NZ, NC, NT, NP, M) log10 abundances, padded
                            # to the full master species list
    logZ_grid: Any          # (NZ,)
    CO_grid: Any            # (NC,)
    ln_cia_stack: Any       # (K, NT, L): ln(CIA data * CIA_DATA_SCALE)
    cia_idx1: Any           # (K,) int32
    cia_idx2: Any           # (K,) int32
    ln_hminus_k: Any        # (NT, L): ln(H- k(T, lambda) / k_B)
    stellar_temps: Any      # (NS,)
    stellar_spectra: Any    # (NS, L)
    orig_lambda_grid: Any   # (L0,) full-resolution grid
    orig_stellar_spectra: Any  # (NS, L0)
    exp3_x: Any             # (NE,) tau values for the E3 lookup table
    exp3_y: Any             # (NE,)
    bterm_x: Any            # (NB,) tau values for the bottom-boundary term
    bterm_y: Any            # (NB,) tau^2 E1(tau) - tau e^-tau + e^-tau
    bin_idx: Any            # (B, W) int32 gather indices per bin, or None
    bin_w: Any              # (B, W) 1/0 weights (0 marks padding), or None


class ForwardConfig(NamedTuple):
    """Static (hashable) configuration; changing any field recompiles."""
    n_layers: int            # number of levels in the T/P profile
    abund_mode: str          # 'eq', 'vmr', or 'custom'
    gas_master_idx: tuple    # master indices of fit gases ('vmr' mode)
    ch4_idx: int
    el_idx: int
    h_idx: int
    add_gas: bool
    add_hminus: bool
    add_scattering: bool
    add_collisional: bool
    sort_layers: bool        # non-isothermal profile: sort layers by cell
    use_mie: bool
    has_t_star: bool
    stellar_in_grid: bool    # PHOENIX grid interp vs blackbody (host-known)
    has_spots: bool
    has_surface: bool = False
    surface_temp_given: bool = False


class ForwardInputs(NamedTuple):
    """Per-call inputs.  The small dense arrays (scalars, ints, T/P profiles,
    masks) are packed host-side into the single `packed` vector so each call
    makes one host-to-device transfer instead of many; the cores unpack it
    (cheap, fusable slices) via `unpack_inputs`.  Layout:
    [scalars (SC_N_SCALARS) | ints-as-floats (IX_N_INTS) | T_profile (N) |
     P_profile (N) | shell_mask (N-1) | opac_mask (S)]."""
    packed: Any             # see layout above
    vmrs: Any = None        # (n_gases,) for 'vmr' mode
    custom_log_abund: Any = None  # (N, M) per-layer, for 'custom' mode
    eff_xsec: Any = None    # (L,) Mie effective cross sections
    rh_binned: Any = None   # (L,) surface hemispheric reflectance
    rh_orig: Any = None     # (L0,)
    crust_flux: Any = None  # (NC2,)
    crust_T: Any = None     # (NC2,)


class UnpackedInputs(NamedTuple):
    scalars: Any            # (SC_N_SCALARS,) float32
    ints: Any               # (IX_N_INTS,) int32
    T_profile: Any          # (N,)
    P_profile: Any          # (N,)
    shell_mask: Any         # (N-1,) 1.0 where the shell is above cloud/surface
    opac_mask: Any          # (S,)
    vmrs: Any = None
    custom_log_abund: Any = None
    eff_xsec: Any = None
    rh_binned: Any = None
    rh_orig: Any = None
    crust_flux: Any = None
    crust_T: Any = None


def unpack_inputs(cfg, pin):
    """Slice the packed per-call vector back into named fields (traced)."""
    v = pin.packed
    n = cfg.n_layers
    o = SC_N_SCALARS
    scalars = v[:o]
    ints = v[o:o + IX_N_INTS].astype(jnp.int32)
    o += IX_N_INTS
    T_profile = v[o:o + n]
    P_profile = v[o + n:o + 2 * n]
    shell_mask = v[o + 2 * n:o + 3 * n - 1]
    opac_mask = v[o + 3 * n - 1:]
    return UnpackedInputs(scalars, ints, T_profile, P_profile, shell_mask,
                          opac_mask, pin.vmrs, pin.custom_log_abund,
                          pin.eff_xsec, pin.rh_binned, pin.rh_orig,
                          pin.crust_flux, pin.crust_T)


class AtmosphereOutputs(NamedTuple):
    radii: Any
    dr: Any
    mu_profile: Any
    atm_abund: Any          # (N, M)
    coeff_perm: Any         # (L, N) absorption coefficients, transposed and
                            # (when perm is not None) column-permuted
    perm: Any               # (N,) layer sort order, or None
    inv_perm: Any           # (N,) inverse permutation, or None
    anchor: Any             # (L,) reduce co-output anchoring the opacity
                            # fusion (see _opacity); must stay live
    unbound: Any            # bool scalar

    @property
    def absorption_coeff_atm(self):
        """(N, L) array in original layer order, for full_output consumers."""
        coeff_T = self.coeff_perm if self.perm is None \
            else self.coeff_perm[:, self.inv_perm]
        return coeff_T.T


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


def _layer_log_abundances(cfg, data, sc, inp):
    """Returns (N, M) log10 abundances of every master species at each
    atmospheric layer.  Everything downstream works on this per-layer (1D)
    grid; the equilibrium-chemistry (T, P) grid only appears here, where it
    is interpolated onto the layers."""
    n_master = data.masses.shape[0]
    N = inp.T_profile.shape[0]

    if cfg.abund_mode == "eq":
        la_grid = regular_grid_interp(
            data.logZ_grid, data.CO_grid, data.log_abund_grid,
            sc[SC_LOGZ], sc[SC_CO])                     # (NT, NP, M)
        la = regular_grid_interp(
            data.T_grid, data.log10_P_grid, la_grid,
            inp.T_profile, jnp.log10(inp.P_profile))    # (N, M)
        if cfg.ch4_idx >= 0:
            la = la.at[:, cfg.ch4_idx].add(sc[SC_LOG_CH4])
    elif cfg.abund_mode == "vmr":
        la = jnp.full((N, n_master), LOG_MIN_ABUND, dtype=jnp.float32)
        idx = jnp.asarray(cfg.gas_master_idx, dtype=jnp.int32)
        vals = jnp.broadcast_to(jnp.log10(inp.vmrs)[None, :],
                                (N, len(cfg.gas_master_idx)))
        la = la.at[:, idx].set(vals)
    else:
        la = inp.custom_log_abund                       # (N, M)

    la = jnp.maximum(la, sc[SC_LOG_MIN_ABUND])

    # Quenching: above the quench point (P <= P_quench), hold every species
    # at its abundance at P_quench, interpolated along the profile itself
    k, f = fractional_index(sc[SC_LOG10_P_QUENCH], jnp.log10(inp.P_profile))
    quench_la = la[k] * (1 - f) + la[k + 1] * f         # (M,)
    la = jnp.where((inp.P_profile <= sc[SC_P_QUENCH])[:, None],
                   quench_la[None, :], la)
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
    k, f = fractional_index(ln_ref, ln_P)
    T_ref = T_profile[k] + f * (T_profile[k + 1] - T_profile[k])
    mu_ref = mu_profile[k] + f * (mu_profile[k + 1] - mu_profile[k])
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


def _opacity(cfg, data, sc, inp, atm_abund, T_profile, P_profile):
    """Per-layer absorption coefficients, computed directly at each layer's
    (T, P) instead of on the 2D opacity grid: each opacity source is
    interpolated from its data grid onto the layers (log cross sections,
    linear in 1/T and ln P -- the same coordinates the grid version used) and
    combined with the per-layer abundances.

    Returns the TRANSPOSED coefficients (L, N).  This shape is deliberate,
    and worth 2x on GPU: with layers innermost, consecutive threads of the
    fused opacity kernel share the same wavelength and read the same handful
    of opacity-grid cells (broadcast/cache hits), and consecutive thread
    blocks stream through the wavelength axis so each grid cell is fetched
    from DRAM about once.  With wavelength innermost (an (N, L) output), the
    whole bracketing-row working set (tens to hundreds of MB) is re-read for
    every layer, which thrashes L2.  Both consumers (the transit path-length
    matmul and the eclipse cumulative sum) want (L, N) anyway.

    The species loop is unrolled for the same reason: each (L, N) term fuses
    its 4 corner gathers, the exp, and the abundance weighting into one
    elementwise accumulation, so no (S, L, N) intermediate is materialized.

    When the T/P profile is not isothermal (cfg.sort_layers), the layers are
    additionally processed in order of their bracketing grid cell:
    neighboring GPU threads then read the same grid rows instead of up-to-32
    scattered ones, which is worth ~3x on profiles with strong temperature
    gradients.  The permutation only reorders per-layer computations, so
    results are unchanged; the columns are left in sorted order (cheaper for
    the matmul consumers, which fold the permutation into their small
    matrices) along with the permutation arrays.

    Returns (coeff_perm, perm, inv_perm, anchor).  `anchor` is a reduce over
    layers emitted from the same fusion: reduce-rooted fusions iterate with
    the layer axis innermost per thread, keeping each layer run's grid cells
    in registers, which is another ~2x over a plain materializing fusion.
    The anchor output must be kept live (it is protected by an
    optimization_barrier together with the coefficients).
    """
    L = data.lambda_grid.shape[0]
    N = T_profile.shape[0]
    n_atm = P_profile / (k_B * T_profile)                          # (N,)

    # Bracketing grid rows/columns and interpolation weights for each layer.
    # The T weight is recomputed in 1/T (the interpolation coordinate).
    t_lo, _ = fractional_index(T_profile, data.T_grid)
    inv_T_lo = data.inv_T_grid[t_lo]
    a = (1.0 / T_profile - inv_T_lo) / (data.inv_T_grid[t_lo + 1] - inv_T_lo)
    a = jnp.clip(a, 0.0, 1.0)                                      # (N,)
    p_lo, b = fractional_index(jnp.log(P_profile), data.ln_P_grid)

    # The accumulator is seeded with the scattering term rather than zeros,
    # and -- in the sorted case -- that seed is passed through a column
    # gather.  Both details steer XLA's fusion/layout decisions for the whole
    # accumulation chain; with a plain zero seed the fused kernel re-reads
    # every species' grid rows per output element and runs 2-5x slower.
    if cfg.add_scattering:
        if cfg.use_mie:
            factor, slope, ref_um = 1.0, 4.0, 1.0
        else:
            factor = sc[SC_SCAT_FACTOR]
            slope = sc[SC_SCAT_SLOPE]
            ref_um = sc[SC_SCAT_REF_UM]
        sum_pol = atm_abund @ data.pol_sqr                         # (N,)
        pow_term = ref_um ** (slope - 4) / data.lambda_um ** slope  # (L,)
        # materialize: fused into the (L, N) coeff kernel, the expensive
        # pow would be recomputed for every layer
        pow_term = lax.optimization_barrier(pow_term)
        seed = (factor * RAYLEIGH_PREF) * \
            (n_atm * sum_pol)[None, :] * pow_term[:, None]
    else:
        seed = lax.optimization_barrier(
            jnp.zeros((L, N), dtype=jnp.float32))

    if cfg.sort_layers:
        NP = data.P_grid.shape[0]
        perm = jnp.argsort(t_lo * NP + p_lo)
        inv_perm = jnp.argsort(perm)
        t_lo = t_lo[perm]
        a = a[perm]
        p_lo = p_lo[perm]
        b = b[perm]
        n_atm = n_atm[perm]
        atm_abund = atm_abund[perm]
        T_profile = T_profile[perm]
        P_profile = P_profile[perm]
        seed = seed[:, perm]
    else:
        perm = None
        inv_perm = None

    coeff_T = seed

    if cfg.add_gas:
        aw = a[None, :]
        bw = b[None, :]
        w00 = (1 - aw) * (1 - bw)
        w01 = (1 - aw) * bw
        w10 = aw * (1 - bw)
        w11 = aw * bw
        gas_ab = atm_abund[:, data.opac_master_idx] * \
            inp.opac_mask[None, :] * n_atm[:, None]                # (N, S)
        for s in range(data.ln_xsec_stack.shape[0]):
            ln_sig = data.ln_xsec_stack[s]
            ln_sig_atm = ln_sig[t_lo, p_lo].T * w00 + \
                ln_sig[t_lo, p_lo + 1].T * w01 + \
                ln_sig[t_lo + 1, p_lo].T * w10 + \
                ln_sig[t_lo + 1, p_lo + 1].T * w11                 # (L, N)
            ln_sig_atm = jnp.maximum(ln_sig_atm, sc[SC_LN_MIN_XSEC])
            coeff_T += jnp.exp(ln_sig_atm) * gas_ab[:, s][None, :]

    if cfg.add_hminus:
        a1 = a[None, :]
        ln_k_atm = data.ln_hminus_k[t_lo].T * (1 - a1) + \
            data.ln_hminus_k[t_lo + 1].T * a1                      # (L, N)
        w = atm_abund[:, cfg.el_idx] * atm_abund[:, cfg.h_idx] * \
            P_profile ** 2 / T_profile                             # (N,)
        coeff_T += jnp.exp(ln_k_atm) * w[None, :]

    if cfg.use_mie:
        n_mie = sc[SC_NUM_DEN] * \
            (P_profile / sc[SC_MIE_REF_P]) ** (1.0 / sc[SC_FSH])   # (N,)
        coeff_T += n_mie[None, :] * inp.eff_xsec[:, None]

    if cfg.add_collisional:
        a1 = a[None, :]
        ns = n_atm * N_SCALE
        wc = atm_abund[:, data.cia_idx1] * atm_abund[:, data.cia_idx2] * \
            (ns * ns)[:, None]                                     # (N, K)
        for k in range(data.ln_cia_stack.shape[0]):
            ln_cia = data.ln_cia_stack[k]
            ln_cia_atm = ln_cia[t_lo].T * (1 - a1) + \
                ln_cia[t_lo + 1].T * a1                            # (L, N)
            coeff_T += jnp.exp(ln_cia_atm) * wc[:, k][None, :]

    anchor = jnp.sum(coeff_T, axis=1)
    coeff_T, anchor = lax.optimization_barrier((coeff_T, anchor))
    return coeff_T, perm, inv_perm, anchor


def _compute_atmosphere(cfg, data, inp):
    sc = inp.scalars
    T_profile = inp.T_profile
    P_profile = inp.P_profile

    la_atm = _layer_log_abundances(cfg, data, sc, inp)   # (N, M)
    atm_abund = 10.0 ** la_atm
    mu_profile = atm_abund @ data.masses                 # (N,)

    radii, dr, unbound = _hydrostatic(sc, P_profile, T_profile, mu_profile)
    coeff_perm, perm, inv_perm, anchor = _opacity(
        cfg, data, sc, inp, atm_abund, T_profile, P_profile)
    return AtmosphereOutputs(radii, dr, mu_profile, atm_abund, coeff_perm,
                             perm, inv_perm, anchor, unbound)


def _planck(lambda_grid, T):
    """pi-free Planck spectral radiance B_lambda(T); T may broadcast."""
    return TWO_H_C_SQR / lambda_grid ** 5 / \
        jnp.expm1(HC_OVER_KB / (lambda_grid * T))


def planck_np(lambda_grid, T):
    """Host (numpy, float64) twin of _planck."""
    lam = np.asarray(lambda_grid)
    return TWO_H_C_SQR / lam ** 5 / np.expm1(HC_OVER_KB / (lam * T))


def _stellar_spectrum(cfg, data, sc, orig=False):
    """Stellar spectrum and spot correction factors on the wavelength grid.
    Whether the PHOENIX grid or a blackbody is used, and whether spots are
    present, are host-known and static, so only the needed branch is traced."""
    lam = data.orig_lambda_grid if orig else data.lambda_grid
    spectra = data.orig_stellar_spectra if orig else data.stellar_spectra
    L = lam.shape[0]
    if not cfg.has_t_star:
        ones = jnp.ones(L, dtype=jnp.float32)
        return ones, ones

    T_star = sc[SC_T_STAR]
    T_spot = sc[SC_T_SPOT]
    f_spot = sc[SC_SPOT_FRAC]

    if cfg.stellar_in_grid:
        unspotted = interp1d(T_star, data.stellar_temps, spectra)
    else:
        unspotted = math.pi * _planck(lam, T_star)
    if not cfg.has_spots:
        return unspotted, jnp.ones(L, dtype=jnp.float32)

    if cfg.stellar_in_grid:
        spot = interp1d(T_spot, data.stellar_temps, spectra)
    else:
        spot = math.pi * _planck(lam, T_spot)
    spectrum = f_spot * spot + (1 - f_spot) * unspotted
    correction_factors = unspotted / spectrum
    return spectrum, correction_factors


def _uniform_log_lookup(x, table_x, table_y, left, right):
    """Linear interpolation of (table_x, table_y) at x, where table_x is
    uniform in log10 (np.logspace): the bracketing segment is found
    analytically instead of by binary search.  The interpolation weight within
    the segment is linear in x, matching jnp.interp; `left`/`right` are the
    values returned outside the table range."""
    n = table_x.shape[0]
    log_x0 = jnp.log10(table_x[0])
    scale = (n - 1) / (jnp.log10(table_x[-1]) - log_x0)
    idx = (jnp.log10(x) - log_x0) * scale
    idx = jnp.clip(idx, 0, n - 2).astype(jnp.int32)
    x0 = table_x[idx]
    frac = (x - x0) / (table_x[idx + 1] - x0)
    y = table_y[idx] * (1 - frac) + table_y[idx + 1] * frac
    y = jnp.where(x < table_x[0], left, y)
    return jnp.where(x > table_x[-1], right, y)


def _get_dl(radii):
    """dl[i, j]: distance travelled by a ray with impact parameter radii[j+1]
    between the shells at radii[i+1] and radii[i]. (radii descending)"""
    # (a - b) * (a + b) instead of a**2 - b**2 avoids FP32 cancellation
    a = radii[:, None]        # r_prime (shell), varying along rows
    b = radii[None, 1:]       # ray impact parameters
    sqr_length = jnp.maximum((a - b) * (a + b), 0.0)
    lengths = 2 * jnp.sqrt(sqr_length)
    return lengths[:-1] - lengths[1:]


def _transit_core(cfg, data, pin):
    inp = unpack_inputs(cfg, pin)
    sc = inp.scalars
    atm = _compute_atmosphere(cfg, data, inp)
    Rs = sc[SC_RS]

    # tau_los[l,j] = sum_i 0.5*(k[i]+k[i+1]) * dl[i,j]: rather than averaging
    # the (L, N) coefficients to midpoints (an extra full-size pass), fold the
    # averaging into the small dl matrix: sum_i k[i] * 0.5*(dl[i]+dl[i-1]).
    # The layer sort from _opacity is likewise folded in by permuting the
    # rows of the small matrix instead of un-permuting the coefficients
    dl = _get_dl(atm.radii)                                    # (N-1, N-1)
    pad = jnp.zeros((1, dl.shape[1]), dtype=dl.dtype)
    dl_mid = 0.5 * (jnp.concatenate([dl, pad]) +
                    jnp.concatenate([pad, dl]))                # (N, N-1)
    if atm.perm is not None:
        dl_mid = dl_mid[atm.perm]
    tau_los = atm.coeff_perm @ dl_mid                          # (L, N-1)
    absorption_fraction = -jnp.expm1(-tau_los)

    shell_w = inp.shell_mask * atm.radii[1:] * atm.dr
    r_floor = atm.radii[inp.ints[IX_FLOOR]]
    depths = (r_floor / Rs) ** 2 + \
        2.0 / Rs ** 2 * (absorption_fraction @ shell_w)

    stellar, corr = _stellar_spectrum(cfg, data, sc)
    if data.bin_idx is not None:
        weighted = jnp.sum((depths * corr * stellar)[data.bin_idx] *
                           data.bin_w, axis=1)
        norm = jnp.sum(stellar[data.bin_idx] * data.bin_w, axis=1)
        binned = weighted / norm
    else:
        binned = depths * corr

    return TransitOutputs(binned, depths, stellar, corr, tau_los,
                          absorption_fraction, atm)


def _eclipse_core(cfg, data, pin):
    inp = unpack_inputs(cfg, pin)
    sc = inp.scalars
    atm = _compute_atmosphere(cfg, data, inp)
    Rs = sc[SC_RS]
    Rp = sc[SC_RP]

    # taus[l,j] = sum_{i<=j} 0.5*(k[i]+k[i+1]) * dm[i], with dm = dr * mask:
    # expressed as a matmul with a small triangular weight matrix (instead
    # of a midpoint average + cumsum) so the opacity coefficients are read
    # by a single consumer, in their sorted column order
    intermediate_T = 0.5 * (inp.T_profile[:-1] + inp.T_profile[1:])
    N = cfg.n_layers
    dm = atm.dr * inp.shell_mask                                # (N-1,)
    zero = jnp.zeros(1, dtype=dm.dtype)
    dmn = jnp.concatenate([dm, zero])                           # dm[n]
    dmp = jnp.concatenate([zero, dm])                           # dm[n-1]
    n_idx = jnp.arange(N, dtype=jnp.int32)[:, None]
    j_idx = jnp.arange(N - 1, dtype=jnp.int32)[None, :]
    tau_w = 0.5 * (dmn[:, None] * (n_idx <= j_idx) +
                   dmp[:, None] * (n_idx <= j_idx + 1))         # (N, N-1)
    if atm.perm is not None:
        tau_w = tau_w[atm.perm]
    taus = atm.coeff_perm @ tau_w                               # (L, N-1)

    planck = _planck(data.lambda_grid[:, None], intermediate_T[None, :])

    exp3 = _uniform_log_lookup(taus, data.exp3_x, data.exp3_y,
                               left=0.5, right=0.0)
    exp3_padded = jnp.concatenate(
        [jnp.full((taus.shape[0], 1), 0.5, dtype=taus.dtype), exp3], axis=1)
    integrand = planck * jnp.diff(exp3_padded, axis=1)
    fluxes = -2 * math.pi * jnp.sum(integrand, axis=1)

    max_taus = jnp.max(taus, axis=1)
    # tau^2 E1(tau) - tau e^-tau + e^-tau via a float64-precomputed lookup
    # table.  (jax.scipy.special.exp1's iterative implementation can fail to
    # converge in FP32; the limits are exactly 1 as tau->0 and 0 as tau->inf.)
    bottom_term = _uniform_log_lookup(max_taus, data.bterm_x, data.bterm_y,
                                      left=1.0, right=0.0)
    # the deepest included shell is the one above the floor node
    planck_bot = jnp.take(planck, inp.ints[IX_FLOOR] - 1, axis=1)

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
            ci, cf = fractional_index(irrad, inp.crust_flux)
            surface_temp = inp.crust_T[ci] + \
                cf * (inp.crust_T[ci + 1] - inp.crust_T[ci])
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

    if data.bin_idx is not None:
        photon_w = stellar * data.lambda_grid  # proportional to photon flux
        weighted = jnp.sum((depths * photon_w)[data.bin_idx] * data.bin_w,
                           axis=1)
        norm = jnp.sum(photon_w[data.bin_idx] * data.bin_w, axis=1)
        binned = weighted / norm
    else:
        binned = depths

    return EclipseOutputs(binned, depths, fluxes, stellar, taus, integrand,
                          photosphere_radii, surface_temp, irrad, atm)


def _transit_depths_only(cfg, data, pin):
    """Depths-only variant: returning just the small arrays lets XLA skip
    materializing the large diagnostic outputs (tau_los, absorption_fraction,
    absorption_coeff_atm, ...).  The unbound flag and one element of the
    opacity-fusion anchor (which must stay live; see _opacity) are appended
    to the depths so one device-to-host transfer returns everything."""
    out = _transit_core(cfg, data, pin)
    flag = jnp.where(out.atm.unbound, 1.0, 0.0)
    # keep the flag's reduction chain out of the output-concatenate fusion:
    # fused there it runs as a single-threaded scalar epilogue (tens of us)
    flag = lax.optimization_barrier(flag)
    return jnp.concatenate([out.binned_depths, flag[None],
                            out.atm.anchor[:1]])


def _eclipse_depths_only(cfg, data, pin):
    """Depths + [unbound flag, irradiation, anchor] in one output array."""
    out = _eclipse_core(cfg, data, pin)
    flag = jnp.where(out.atm.unbound, 1.0, 0.0)
    tail = lax.optimization_barrier(
        jnp.stack([flag, jnp.asarray(out.irrad, dtype=jnp.float32)]))
    return jnp.concatenate([out.binned_depths, tail, out.atm.anchor[:1]])


transit_core = jax.jit(_transit_core, static_argnums=0)
eclipse_core = jax.jit(_eclipse_core, static_argnums=0)
transit_depths_core = jax.jit(_transit_depths_only, static_argnums=0)
eclipse_depths_core = jax.jit(_eclipse_depths_only, static_argnums=0)
