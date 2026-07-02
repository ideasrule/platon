from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.special
import jax.numpy as jnp

from . import _forward_model as fm
from ._forward_model import DeviceData, planck_np
from ._interpolator_3D import interp1d_np
from ._hist import get_num_bins
from ._loader import load_dict_from_pickle, load_numpy
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from .constants import k_B
from ._get_data import get_data_if_needed
from ._mie_cache import MieCache
from .errors import AtmosphereError

# Data caches shared between calculator instances, so that constructing many
# calculators doesn't repeatedly read gigabytes from disk or duplicate arrays
# on the GPU.  All cached arrays are immutable.  The device cache is a small
# LRU: each binned entry can hold gigabytes of GPU memory, so old bin sets
# must be evicted (live calculators keep their own references and are
# unaffected by eviction; only cross-instance sharing is lost).
_RAW_CACHE = {}
_LOG_ABUND_CACHE = {}
_DEVICE_CACHE = {}
_DEVICE_CACHE_MAX_ENTRIES = 8


def _interp_rows_to(lambda_target, lambda_source, data):
    """Interpolate data (..., len(lambda_source)) onto lambda_target."""
    return np.array([np.interp(lambda_target, lambda_source, row)
                     for row in data])


def _compute_h_minus_k(T_grid, wavelengths_m):
    """John (1988) H- bound-free + free-free absorption k(T, lambda), in
    m^4/N, for all temperatures at once: returns (len(T_grid), L).  Runs in
    float64 on the host (precomputed once per data load); the
    wavelength-dependent factors are computed once and shared by all T."""
    T = np.asarray(T_grid, dtype=np.float64)[:, np.newaxis]
    wavelengths = 1e6 * np.asarray(wavelengths_m, dtype=np.float64)
    alpha = 14391
    lambda_0 = 1.6419

    k_bf = np.zeros((len(T), len(wavelengths)))
    cond = wavelengths < lambda_0
    C = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
    f_lambda = np.sum([C[i - 1] * (1 / wavelengths[cond] - 1 / lambda_0)**((i - 1) / 2)
                       for i in range(1, 7)], axis=0)
    sigma = 1e-18 * wavelengths[cond]**3 * \
        (1 / wavelengths[cond] - 1 / lambda_0)**1.5 * f_lambda
    k_bf[:, cond] = 0.75 * T**-2.5 * np.exp(alpha / lambda_0 / T) * \
        (1 - np.exp(-alpha / wavelengths[cond] / T)) * sigma

    k_ff = np.zeros((len(T), len(wavelengths)))
    mid = np.logical_and(wavelengths > 0.1823, wavelengths < 0.3645)
    red = wavelengths > 0.3645

    ff_matrix_red = np.array([
        [0, 0, 0, 0, 0, 0],
        [2483.346, 285.827, -2054.291, 2827.776, -1341.537, 208.952],
        [-3449.889, -1158.382, 8746.523, -11485.632, 5303.609, -812.939],
        [2200.04, 2427.719, -13651.105, 16755.524, -7510.494, 1132.738],
        [-696.271, -1841.4, 8624.97, -10051.53, 4400.067, -655.02],
        [88.283, 444.517, -1863.864, 2095.288, -901.788, 132.985]])
    ff_matrix_mid = np.array([
        [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
        [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.36],
        [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
        [115.5291, 169.6374, -245.649, 114.243, -21.9972, 1.5097],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])

    A_mid = np.array([wavelengths[mid]**i for i in (2, 0, -1, -2, -3, -4)]).T
    A_red = np.array([wavelengths[red]**i for i in (2, 0, -1, -2, -3, -4)]).T
    # (T, 6) temperature weights x (6, L) wavelength basis
    T_weights = np.hstack([1e-29 * (5040 / T)**((n + 1) / 2) for n in range(1, 7)])
    k_ff[:, mid] = T_weights @ ff_matrix_mid @ A_mid.T
    k_ff[:, red] = T_weights @ ff_matrix_red @ A_red.T

    # 1e-3 to convert from cm^4/dyne to m^4/N
    return (k_bf + k_ff) * 1e-3


def _load_raw(method, include_opacities, downsample):
    """Load all wavelength-dependent data at full resolution (CPU, float32)."""
    # sorted: the same opacity set in a different order must share one entry
    key = (method, tuple(sorted(include_opacities)), downsample)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    basedir = Path(__file__).resolve().parent
    absorption_files, mass_data, polarizability_data = read_species_data(
        basedir / "data/Absorption", basedir / "data/species_info",
        method, include_opacities)

    lambda_full = load_numpy("data/wavelengths.npy")[::downsample]
    low_res_lambdas = load_numpy("data/low_res_lambdas.npy")

    master_names = list(mass_data.keys())
    master_index = {name: i for i, name in enumerate(master_names)}
    masses = np.array([mass_data[name] for name in master_names], np.float32)
    pol_sqr = np.array(
        [(polarizability_data.get(name, 0.0) * fm.POL_SCALE)**2
         for name in master_names], np.float32)

    # Load each opacity file straight into a preallocated float32 stack
    # (float32 is the working precision of the JAX pipeline); avoids a
    # second multi-GB copy from np.stack
    opac_names = list(absorption_files.keys())
    NT_g, NP_g = 40, 13
    abs_stack = np.empty((len(opac_names), NT_g, NP_g, len(lambda_full)),
                         np.float32)
    for i, name in enumerate(opac_names):
        raw = np.load(absorption_files[name], mmap_mode="r")
        abs_stack[i] = raw[:, :, ::downsample]
    opac_master_idx = np.array(
        [master_index[name] for name in opac_names], np.int32)

    collisional = load_dict_from_pickle("data/collisional_absorption.pkl")
    cia_pairs = [(s1, s2) for (s1, s2) in collisional
                 if s1 in master_index and s2 in master_index]
    if len(cia_pairs) > 0:
        cia_stack = np.stack(
            [(_interp_rows_to(lambda_full, low_res_lambdas,
                              collisional[pair]) * fm.CIA_DATA_SCALE
              ).astype(np.float32) for pair in cia_pairs])
    else:
        cia_stack = np.zeros((0, 40, len(lambda_full)), np.float32)
    cia_idx1 = np.array([master_index[s1] for s1, _ in cia_pairs], np.int32)
    cia_idx2 = np.array([master_index[s2] for _, s2 in cia_pairs], np.int32)

    P_grid = load_numpy("data/pressures.npy").astype(np.float64)
    T_grid = load_numpy("data/temperatures.npy").astype(np.float64)

    hminus_k = (_compute_h_minus_k(T_grid, lambda_full) / k_B
                ).astype(np.float32)

    stellar_dict = load_dict_from_pickle("data/stellar_spectra.pkl")
    stellar_temps = np.asarray(stellar_dict["temperatures"], np.float64)
    stellar_spectra = _interp_rows_to(
        lambda_full, low_res_lambdas,
        np.asarray(stellar_dict["spectra"])).astype(np.float32)

    exp3_x = np.logspace(-6, 3, 1000)
    exp3_y = scipy.special.expn(3, exp3_x)

    # Lookup table for tau^2 E1(tau) - tau e^-tau + e^-tau (float64 host
    # computation; jax's exp1 is unreliable in FP32)
    bterm_x = np.logspace(-6, 2.5, 10000)
    bterm_y = bterm_x**2 * scipy.special.exp1(bterm_x) - \
        bterm_x * np.exp(-bterm_x) + np.exp(-bterm_x)

    raw = dict(
        key=key,
        lambda_full=lambda_full,
        low_res_lambdas=low_res_lambdas,
        master_names=master_names, master_index=master_index,
        masses=masses, pol_sqr=pol_sqr,
        opac_names=opac_names, abs_stack=abs_stack,
        opac_master_idx=opac_master_idx,
        cia_stack=cia_stack,
        cia_idx1=cia_idx1, cia_idx2=cia_idx2,
        P_grid=P_grid, T_grid=T_grid,
        hminus_k=hminus_k,
        stellar_temps=stellar_temps, stellar_spectra=stellar_spectra,
        exp3_x=exp3_x.astype(np.float32), exp3_y=exp3_y.astype(np.float32),
        bterm_x=bterm_x.astype(np.float32), bterm_y=bterm_y.astype(np.float32),
    )
    _RAW_CACHE[key] = raw
    return raw


def _get_log_abund_grid(include_condensation, abundance_getter, master_index):
    """(NZ, NC, S_eq, NT, NP) float32 log10 abundance grid, floored at -99.
    Derived from the AbundanceGetter's grid (already loaded from disk)."""
    if include_condensation in _LOG_ABUND_CACHE:
        return _LOG_ABUND_CACHE[include_condensation]
    log_grid = abundance_getter.log_abundances.astype(np.float32)
    eq_master_idx = np.array(
        [master_index[s] for s in abundance_getter.included_species], np.int32)
    result = (log_grid, eq_master_idx)
    _LOG_ABUND_CACHE[include_condensation] = result
    return result


class AtmosphereSolver:
    def __init__(self, include_condensation=True, ref_pressure=1e5,
                 method='xsec', include_opacities=[], downsample=1):
        if method == "ktables":
            raise NotImplementedError(
                "Correlated-k support has been removed from this JAX version "
                "of PLATON; use method='xsec'")

        get_data_if_needed()

        self.raw = _load_raw(method, include_opacities, downsample)
        self.include_condensation = include_condensation

        self.orig_lambda_grid = np.array(self.raw["lambda_full"])
        self.lambda_grid = np.array(self.raw["lambda_full"])

        self.P_grid = self.raw["P_grid"]
        self.T_grid = self.raw["T_grid"]
        self.N_lambda = len(self.lambda_grid)
        self.N_T = len(self.T_grid)
        self.N_P = len(self.P_grid)

        self.stellar_spectra_temps = self.raw["stellar_temps"]

        self.wavelength_bins = None

        self.abundance_getter = AbundanceGetter(include_condensation)
        self.min_temperature = max(self.T_grid.min(),
                                   self.abundance_getter.min_temperature)
        self.max_temperature = self.T_grid.max()

        self.ref_pressure = ref_pressure
        self._mie_cache = MieCache()
        self._filtered_cross_secs = {}   # (species, sigma) -> smoothed array

        self.all_cross_secs = load_dict_from_pickle("data/all_cross_secs.pkl")
        self.all_radii = load_numpy("data/mie_radii.npy")
        diffs = np.diff(np.log(self.all_radii))
        self.d_ln_radii = np.median(diffs)
        assert np.allclose(diffs, self.d_ln_radii)

        self.low_res_lambdas = self.raw["low_res_lambdas"]

        self.master_names = self.raw["master_names"]
        self.master_index = self.raw["master_index"]

        self._lambda_cond = None      # boolean mask into lambda_full
        self._bin_info = None
        self._device_data = None

    # ------------------------------------------------------------------
    # Wavelength binning
    # ------------------------------------------------------------------
    def get_lambda_grid(self):
        return np.array(self.lambda_grid)

    def change_wavelength_bins(self, bins):
        """Specify wavelength bins, instead of using the full wavelength grid
        in self.lambda_grid.  This makes the code much faster, as
        `compute_depths` will only compute depths at wavelengths that fall
        within a bin.

        Parameters
        ----------
        bins : array_like, shape (N,2)
            Wavelength bins, where bins[i][0] is the start wavelength and
            bins[i][1] is the end wavelength for bin i. If bins is None, resets
            the calculator to its unbinned state.
        """
        if self.wavelength_bins is not None:
            # Reset to the unbinned state before applying the new bins
            self.lambda_grid = np.array(self.orig_lambda_grid)
            self.N_lambda = len(self.lambda_grid)
            self.wavelength_bins = None
            self._lambda_cond = None
            self._bin_info = None
            self._device_data = None

        if bins is None:
            return
        bins = np.asarray(bins, dtype=np.float64)

        full = self.orig_lambda_grid
        for start, end in bins:
            if start < full.min() or start > full.max() \
               or end < full.min() or end > full.max():
                raise ValueError(
                    "Invalid wavelength bin: {}-{} meters".format(start, end))
            num_points = np.sum(np.logical_and(full > start, full < end))
            if num_points == 0:
                raise ValueError(
                    "Wavelength bin too narrow: {}-{} meters".format(start, end))
            if num_points <= 5:
                print("WARNING: only {} points in {}-{} m bin. Results will "
                      "be inaccurate".format(num_points, start, end))

        self.wavelength_bins = bins

        cond = np.any([np.logical_and(full > start, full < end)
                       for (start, end) in bins], axis=0)
        self._lambda_cond = cond
        self.lambda_grid = full[cond]
        self.N_lambda = len(self.lambda_grid)
        self._bin_info = self._compute_bin_info(bins)
        self._device_data = None

    def _compute_bin_info(self, bins):
        """Precompute per-bin index ranges, the averaging matrix, and bin
        center wavelengths.  On the sorted wavelength grid, searchsorted
        [l:r) selects exactly the points with start <= lambda < end."""
        lam = self.lambda_grid
        B = len(bins)
        bin_mat = np.zeros((B, len(lam)), dtype=np.float32)
        bin_wavelengths = np.zeros(B)
        bin_ranges = []
        for i, (start, end) in enumerate(bins):
            l = np.searchsorted(lam, start)
            r = np.searchsorted(lam, end)
            bin_mat[i, l:r] = 1
            bin_wavelengths[i] = np.mean(lam[l:r])
            bin_ranges.append((l, r))
        return dict(bin_mat=bin_mat, bin_wavelengths=bin_wavelengths,
                    bin_ranges=bin_ranges)

    # ------------------------------------------------------------------
    # Device data
    # ------------------------------------------------------------------
    def device_data(self):
        if self._device_data is not None:
            return self._device_data

        bins_key = None if self.wavelength_bins is None \
            else self.wavelength_bins.tobytes()
        key = (self.raw["key"], self.include_condensation, bins_key)
        if key in _DEVICE_CACHE:
            # Move to the end so the LRU eviction below treats it as fresh
            self._device_data = _DEVICE_CACHE.pop(key)
            _DEVICE_CACHE[key] = self._device_data
            return self._device_data

        raw = self.raw
        cond = self._lambda_cond
        if cond is None:
            cond = slice(None)

        log_abund_grid, eq_master_idx = _get_log_abund_grid(
            self.include_condensation, self.abundance_getter,
            self.master_index)

        lam = self.lambda_grid
        dd = DeviceData(
            lambda_grid=jnp.asarray(lam, dtype=jnp.float32),
            lambda_um=jnp.asarray(lam * 1e6, dtype=jnp.float32),
            T_grid=jnp.asarray(raw["T_grid"], dtype=jnp.float32),
            P_grid=jnp.asarray(raw["P_grid"], dtype=jnp.float32),
            ln_P_grid=jnp.asarray(np.log(raw["P_grid"]), dtype=jnp.float32),
            log10_P_grid=jnp.asarray(np.log10(raw["P_grid"]), dtype=jnp.float32),
            abs_stack=jnp.asarray(raw["abs_stack"][:, :, :, cond]),
            opac_master_idx=jnp.asarray(raw["opac_master_idx"]),
            masses=jnp.asarray(raw["masses"]),
            pol_sqr=jnp.asarray(raw["pol_sqr"]),
            log_abund_grid=jnp.asarray(log_abund_grid),
            logZ_grid=jnp.asarray(self.abundance_getter.logZs, dtype=jnp.float32),
            CO_grid=jnp.asarray(self.abundance_getter.CO_ratios, dtype=jnp.float32),
            eq_master_idx=jnp.asarray(eq_master_idx),
            cia_stack=jnp.asarray(raw["cia_stack"][:, :, cond]),
            cia_idx1=jnp.asarray(raw["cia_idx1"]),
            cia_idx2=jnp.asarray(raw["cia_idx2"]),
            hminus_k_over_kB=jnp.asarray(raw["hminus_k"][:, cond]),
            stellar_temps=jnp.asarray(raw["stellar_temps"], dtype=jnp.float32),
            stellar_spectra=jnp.asarray(raw["stellar_spectra"][:, cond]),
            orig_lambda_grid=jnp.asarray(raw["lambda_full"], dtype=jnp.float32),
            orig_stellar_spectra=jnp.asarray(raw["stellar_spectra"]),
            exp3_x=jnp.asarray(raw["exp3_x"]),
            exp3_y=jnp.asarray(raw["exp3_y"]),
            bterm_x=jnp.asarray(raw["bterm_x"]),
            bterm_y=jnp.asarray(raw["bterm_y"]),
            bin_mat=None if self._bin_info is None
                else jnp.asarray(self._bin_info["bin_mat"]),
        )
        _DEVICE_CACHE[key] = dd
        while len(_DEVICE_CACHE) > _DEVICE_CACHE_MAX_ENTRIES:
            _DEVICE_CACHE.pop(next(iter(_DEVICE_CACHE)))
        self._device_data = dd
        return dd

    # ------------------------------------------------------------------
    # Host-side helpers
    # ------------------------------------------------------------------
    def get_t0(self, T_profile):
        """First T-grid row of the 2-row slice bracketing an isothermal T."""
        T = float(np.min(T_profile))
        return int(np.clip(np.searchsorted(self.T_grid, T, side="right") - 1,
                           0, self.N_T - 2))

    def get_above_info(self, P_profile, bot_pressure):
        """Node/shell masks for the region above the cloud deck / surface."""
        above_nodes = np.asarray(P_profile) < bot_pressure
        if not above_nodes.any():
            raise AtmosphereError("Entire atmosphere is below the clouds")
        n_above = int(above_nodes.sum())
        shell_mask = above_nodes[1:].astype(np.float32)
        return n_above, shell_mask

    def get_mie_ref_pressure(self, P_profile, bot_pressure):
        """First pressure-grid point at or below the deepest visible level
        (matching the legacy grid-truncation behavior), used as the reference
        for the Mie particle density profile."""
        P_above = np.asarray(P_profile)[np.asarray(P_profile) < bot_pressure]
        target = min(P_above.max(), bot_pressure)
        idx = np.searchsorted(self.P_grid, target)
        idx = min(idx, self.N_P - 1)
        return float(self.P_grid[idx])

    def get_quench_T(self, P_profile, T_profile, P_quench):
        return float(np.interp(np.log(P_quench), np.log(P_profile), T_profile))

    def custom_abundances_to_log_master(self, custom_abundances):
        """Convert a species -> (N_T, N_P) abundance dict to a master-species
        log10 array."""
        unknown = [key for key in custom_abundances
                   if key not in self.master_index]
        if unknown:
            raise ValueError(
                "custom_abundances contains unknown species: {}".format(unknown))
        result = np.full((len(self.master_names), self.N_T, self.N_P),
                         fm.LOG_MIN_ABUND, dtype=np.float32)
        for key, value in custom_abundances.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(
                    "custom_abundances must map species names to arrays")
            if value.shape != (self.N_T, self.N_P):
                raise ValueError(
                    "custom_abundances has array of invalid size")
            with np.errstate(divide="ignore", invalid="ignore"):
                logv = np.log10(value.astype(np.float64))
            logv = np.nan_to_num(logv, nan=fm.LOG_MIN_ABUND,
                                 neginf=fm.LOG_MIN_ABUND)
            result[self.master_index[key]] = logv
        return result

    def _validate_params(self, T_profile, logZ, CO_ratio, cloudtop_pressure):
        T_profile = np.atleast_1d(np.asarray(T_profile, dtype=np.float64))
        if T_profile.min() < self.min_temperature or \
           T_profile.max() > self.max_temperature:
            raise AtmosphereError("Invalid temperatures in T/P profile")

        if logZ is not None:
            minimum = float(self.abundance_getter.logZs.min())
            maximum = float(self.abundance_getter.logZs.max())
            if logZ < minimum or logZ > maximum:
                raise ValueError(
                    "logZ {} is out of bounds ({} to {})".format(
                        logZ, minimum, maximum))

        if CO_ratio is not None:
            minimum = float(self.abundance_getter.CO_ratios.min())
            maximum = float(self.abundance_getter.CO_ratios.max())
            if CO_ratio < minimum or CO_ratio > maximum:
                raise ValueError(
                    "C/O ratio {} is out of bounds ({} to {})".format(
                        CO_ratio, minimum, maximum))

        if not np.isinf(cloudtop_pressure):
            minimum = float(self.P_grid.min())
            maximum = float(self.P_grid.max())
            if cloudtop_pressure <= minimum or cloudtop_pressure > maximum:
                raise ValueError(
                    "Cloudtop pressure is {} Pa, but must be between {} and "
                    "{} Pa unless it is np.inf".format(
                        cloudtop_pressure, minimum, maximum))

    def get_stellar_spectrum(self, T_star, T_spot, spot_cov_frac,
                             blackbody=False, use_full_lambdas=False):
        """Host (numpy) stellar spectrum, for non-JIT use."""
        if use_full_lambdas:
            lambdas = self.orig_lambda_grid
            stellar_spectra = self.raw["stellar_spectra"]
        else:
            lambdas = self.lambda_grid
            cond = self._lambda_cond
            stellar_spectra = self.raw["stellar_spectra"] if cond is None \
                else self.raw["stellar_spectra"][:, cond]

        if spot_cov_frac is None:
            spot_cov_frac = 0

        if T_spot is None:
            T_spot = T_star

        temps = self.stellar_spectra_temps
        if T_star is None:
            unspotted_spectrum = np.ones(len(lambdas))
            spot_spectrum = np.ones(len(lambdas))
        elif T_star >= temps.min() and T_star <= temps.max() and not blackbody:
            unspotted_spectrum = interp1d_np(T_star, temps, stellar_spectra)
            spot_spectrum = interp1d_np(T_spot, temps, stellar_spectra)
        else:
            unspotted_spectrum = np.pi * planck_np(lambdas, T_star)
            spot_spectrum = np.pi * planck_np(lambdas, T_spot)

        stellar_spectrum = spot_cov_frac * spot_spectrum + \
            (1 - spot_cov_frac) * unspotted_spectrum
        correction_factors = unspotted_spectrum / stellar_spectrum
        return stellar_spectrum, correction_factors

    # ------------------------------------------------------------------
    # Mie scattering (host side; effective cross sections are passed into
    # the JIT core as an input array)
    # ------------------------------------------------------------------
    def get_mie_eff_cross_section(self, ri, part_size, sigma=0.5,
                                  max_zscore=5, num_integral_points=100):
        """Effective extinction cross section vs wavelength for a log-normal
        particle size distribution.  float64, on the host."""
        if isinstance(ri, str):
            if ri not in self.all_cross_secs:
                raise ValueError("Unknown aerosol species: {}".format(ri))
            if sigma <= 0.05:
                raise ValueError(
                    "part_size_std must be > 0.05 for aerosol-species Mie "
                    "scattering (got {})".format(sigma))
            kernel = sigma / self.d_ln_radii
            if (ri, sigma) not in self._filtered_cross_secs:
                self._filtered_cross_secs[(ri, sigma)] = \
                    scipy.ndimage.gaussian_filter(self.all_cross_secs[ri],
                                                  kernel)
            cross_secs = self._filtered_cross_secs[(ri, sigma)]
            if part_size < self.all_radii[3 * int(kernel)] or \
               part_size > self.all_radii[-3 * int(kernel)]:
                raise ValueError("part_size out of bounds: {} m".format(part_size))

            at_radius = interp1d_np(part_size, self.all_radii, cross_secs.T)
            return np.interp(self.lambda_grid, self.low_res_lambdas, at_radius)

        z_scores = -np.logspace(np.log10(0.1), np.log10(max_zscore),
                                int(num_integral_points / 2))
        z_scores = np.append(z_scores[::-1], -z_scores)

        probs = np.exp(-z_scores**2 / 2) / np.sqrt(2 * np.pi)
        radii = part_size * np.exp(z_scores * sigma)
        geometric_cross_section = np.pi * radii**2

        # log(2 pi r / lambda) computed by broadcasting logs (cheaper than
        # taking the log of the full L x n_radii matrix)
        log_dense_xs = (np.log(2 * np.pi * radii)[np.newaxis, :] -
                        np.log(self.lambda_grid)[:, np.newaxis])

        n_bins = get_num_bins(log_dense_xs.flatten())
        log_x_hist = np.histogram(log_dense_xs.flatten(), bins=n_bins)[1]

        Qext_hist = self._mie_cache.get_and_update(ri, np.exp(log_x_hist))
        spl = scipy.interpolate.make_interp_spline(log_x_hist, Qext_hist)
        Qext_intpl = spl(log_dense_xs)
        return np.trapezoid(probs * geometric_cross_section * Qext_intpl,
                            z_scores, axis=1)

    def _get_mie_scattering_absorption(self, P_cond, T_cond, ri, part_size,
                                       frac_scale_height, max_number_density,
                                       sigma=0.5, max_zscore=5,
                                       num_integral_points=100):
        """Backward-compatible host implementation returning absorption
        coefficients of shape (1, sum(P_cond), N_lambda)."""
        eff_cross_section = self.get_mie_eff_cross_section(
            ri, part_size, sigma, max_zscore, num_integral_points)
        P = np.asarray(self.P_grid)[np.asarray(P_cond)]
        n = max_number_density * np.power(P / max(P), 1.0 / frac_scale_height)
        return n[np.newaxis, :, np.newaxis] * \
            eff_cross_section[np.newaxis, np.newaxis, :]
