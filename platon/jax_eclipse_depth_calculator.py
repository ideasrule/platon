"""JAX-accelerated emission (secondary-eclipse) depth calculator.

Drop-in replacement for
:class:`platon.eclipse_depth_calculator.EclipseDepthCalculator` that evaluates
the forward model with the JIT-compiled JAX kernels in ``_jax_eclipse_model``
instead of NumPy/CuPy.  The public API (constructor arguments,
``change_wavelength_bins``, ``compute_depths``) is unchanged, so existing
scripts only need to swap the import::

    from platon.jax_eclipse_depth_calculator import EclipseDepthCalculator

Static arrays (opacities, stellar spectrum, binning weights) are packed once
and reused, and the compiled kernel is cached, so the cost is paid on the first
``compute_depths`` call and amortised over a retrieval.

Options without a JAX implementation — correlated-k, Mie scattering, rocky
surfaces, ``custom_abundances`` — raise
:class:`platon._jax_forward_model.UnsupportedJAXFeatureError`; use the NumPy
calculator for those.
"""

import numpy as np

from . import _cupy_numpy as xp
from ._atmosphere_solver import AtmosphereSolver
from .fit_info import FitInfo
from ._params import _Param
from ._jax_forward_model import UnsupportedJAXFeatureError
from ._jax_eclipse_model import (prepare_jax_eclipse_data,
                                 jax_compute_eclipse_depths)


def _to_numpy(value):
    """Convert a JAX/CuPy array (or nested dict of them) to NumPy."""
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    return np.asarray(value)


class EclipseDepthCalculator:
    def __init__(self, include_condensation=True, method="xsec",
                 include_opacities=["CH4", "CO2", "CO", "H2O", "H2S", "HCN",
                                    "K", "Na", "NH3", "SO2", "TiO", "VO"],
                 downsample=1, surface_library="Paragas"):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        method : string
            Only "xsec" (opacity sampling) is supported by the JAX model.
        include_opacities : list of str
            Molecules whose opacities are loaded.
        downsample : int
            Stride applied to the wavelength grid when loading opacities.
        surface_library : string
            Accepted for API compatibility; rocky surfaces are not implemented
            in the JAX model.
        '''
        if method != "xsec":
            raise UnsupportedJAXFeatureError(
                "The JAX emission model only supports method='xsec'. Use "
                "platon.eclipse_depth_calculator.EclipseDepthCalculator for "
                "correlated-k.")
        self.atm = AtmosphereSolver(include_condensation, method=method,
                                    include_opacities=include_opacities,
                                    downsample=downsample)
        self.surface_library = surface_library
        self.wavelength_bins = None
        self._binned_wavelengths = None
        self._cache_key = None
        self._jax_data = None
        self._jax_fn = None

    # ------------------------------------------------------------------ API

    def change_wavelength_bins(self, bins):
        '''Same functionality as
        :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)
        self.wavelength_bins = None if bins is None else np.asarray(
            xp.cpu(xp.array(bins)), dtype=np.float64)
        self._binned_wavelengths = None
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._cache_key = None
        self._jax_data = None
        self._jax_fn = None

    # -------------------------------------------------------------- helpers

    def _bin_centres(self):
        """Mean wavelength of the grid points falling in each bin."""
        if self._binned_wavelengths is not None:
            return self._binned_wavelengths
        lambda_grid = np.asarray(xp.cpu(self.atm.lambda_grid), dtype=np.float64)
        if self.wavelength_bins is None:
            self._binned_wavelengths = lambda_grid
        else:
            centres = []
            for start, end in self.wavelength_bins:
                mask = np.logical_and(lambda_grid > start, lambda_grid < end)
                centres.append(lambda_grid[mask].mean())
            self._binned_wavelengths = np.array(centres)
        return self._binned_wavelengths

    @staticmethod
    def _build_fit_info(profile_type, vmr_mode, gases, T_star, T_spot,
                        spot_cov_frac, add_H_minus_absorption, ri,
                        add_gas_absorption, add_scattering,
                        add_collisional_absorption, scattering_ref_wavelength,
                        custom_abundances):
        """A minimal FitInfo describing the static configuration.

        ``prepare_jax_data`` reads the model configuration off a FitInfo
        object; nothing is fitted here, so ``fit_param_names`` stays empty and
        every value is a fixed best guess.
        """
        guesses = {
            "profile_type": profile_type,
            "fit_vmr": vmr_mode,
            "fit_clr": False,
            "T_star": T_star,
            "T_spot": T_spot,
            "spot_cov_frac": spot_cov_frac,
            "add_H_minus_absorption": add_H_minus_absorption,
            "add_gas_absorption": add_gas_absorption,
            "add_scattering": add_scattering,
            "add_collisional_absorption": add_collisional_absorption,
            "scattering_ref_wavelength": scattering_ref_wavelength,
            "custom_abundances": custom_abundances,
            "cloud_cov_frac": 1.0,
            "limb_asym": False,
            "log_P_quench": -99,
            "quench_species": None,
            "n": None if ri is None else float(np.real(ri)),
        }
        fit_info = FitInfo(guesses)
        if vmr_mode:
            fit_info.gases = list(gases)
        return fit_info

    def _get_compiled(self, key, fit_info, T_star, T_spot, spot_cov_frac,
                      stellar_blackbody, zero_opacities):
        if self._cache_key == key and self._jax_fn is not None:
            return self._jax_data, self._jax_fn

        import jax

        n_points = (len(self.wavelength_bins)
                    if self.wavelength_bins is not None
                    else len(self.atm.lambda_grid))
        data = prepare_jax_eclipse_data(
            self.atm, self.atm.abundance_getter, self.wavelength_bins,
            T_star, T_spot, spot_cov_frac, stellar_blackbody,
            fit_info, n_points, zero_opacities=zero_opacities)

        @jax.jit
        def _fn(params_dict):
            return jax_compute_eclipse_depths(params_dict, data,
                                              return_full=True)

        self._cache_key = key
        self._jax_data = data
        self._jax_fn = _fn
        return data, _fn

    # ------------------------------------------------------------- main API

    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, T_star, logZ=0, CO_ratio=0.53,
                       CH4_mult=1, gases=None, vmrs=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       stellar_blackbody=False,
                       full_output=False, zero_opacities=[],
                       surface_type=None, semimajor_axis=None,
                       surface_temp=None, surface_pressure=np.inf,
                       offset_eclipse=None, offset_start=None,
                       offset_end=None):
        '''Compute eclipse depths.  Arguments match
        :func:`platon.eclipse_depth_calculator.EclipseDepthCalculator.compute_depths`.

        Returns
        -------
        wavelengths : array of float
            Central wavelengths of each bin, in metres
        depths : array of float
            Eclipse depths at `wavelengths`
        info_dict : dict or None
            Returned if `full_output` is True: unbinned wavelengths and depths,
            planet and stellar spectra, optical depths, the T/P profile, and
            the contribution function.
        '''
        if ri is not None or number_density != 0:
            raise UnsupportedJAXFeatureError(
                "Mie scattering is not implemented in the JAX emission model.")
        if custom_abundances is not None:
            raise UnsupportedJAXFeatureError(
                "custom_abundances is not implemented in the JAX emission "
                "model.")
        if surface_type is not None or np.isfinite(surface_pressure):
            raise UnsupportedJAXFeatureError(
                "Rocky-surface emission is not implemented in the JAX "
                "emission model.")
        if P_quench is not None and P_quench > 1e-50:
            raise UnsupportedJAXFeatureError(
                "Quenching is not wired through the standalone JAX emission "
                "calculator; use the retriever, which configures it from "
                "fit_info.")

        vmr_mode = gases is not None and vmrs is not None
        if vmr_mode:
            if logZ is not None or CO_ratio is not None:
                raise ValueError(
                    "Set logZ=None and CO_ratio=None when passing gases/vmrs")
            gases = list(gases)

        key = (
            "vmr" if vmr_mode else "eq",
            tuple(gases) if vmr_mode else None,
            None if T_star is None else float(T_star),
            None if T_spot is None else float(T_spot),
            None if spot_cov_frac is None else float(spot_cov_frac),
            bool(stellar_blackbody), bool(add_H_minus_absorption),
            bool(add_gas_absorption), bool(add_scattering),
            bool(add_collisional_absorption), float(scattering_ref_wavelength),
            tuple(sorted(zero_opacities)),
            offset_eclipse is not None,
        )
        fit_info = self._build_fit_info(
            "external", vmr_mode, gases, T_star, T_spot, spot_cov_frac,
            add_H_minus_absorption, ri, add_gas_absorption, add_scattering,
            add_collisional_absorption, scattering_ref_wavelength,
            custom_abundances)
        if offset_eclipse is not None:
            fit_info.all_params["offset_eclipse"] = _Param(float(offset_eclipse))
            fit_info.all_params["offset_start"] = _Param(int(offset_start))
            fit_info.all_params["offset_end"] = _Param(int(offset_end))

        data, fn = self._get_compiled(
            key, fit_info, T_star, T_spot, spot_cov_frac, stellar_blackbody,
            zero_opacities)

        T_profile = np.asarray(xp.cpu(t_p_profile.temperatures),
                               dtype=np.float64)
        P_profile = np.asarray(xp.cpu(t_p_profile.pressures), dtype=np.float64)
        expected_P = np.asarray(data["P_profile"], dtype=np.float64)
        if len(T_profile) != len(expected_P):
            raise ValueError(
                "The JAX emission model requires the profile to be sampled on "
                "PLATON's standard {} layer pressure grid; got {} layers."
                .format(len(expected_P), len(T_profile)))
        if not np.allclose(P_profile, expected_P, rtol=1e-4):
            raise ValueError(
                "The JAX emission model requires the standard PLATON pressure "
                "grid (platon.params.MIN_P .. MAX_P).")

        params_dict = {
            "Rp": float(planet_radius),
            "Rs": float(star_radius),
            "Mp": float(planet_mass),
            "T_profile": T_profile.astype(np.float32),
            "log_scatt_factor": float(np.log10(scattering_factor)),
            "scatt_slope": float(scattering_slope),
            "log_cloudtop_P": float(np.log10(cloudtop_pressure)),
        }
        if T_star is not None:
            params_dict["T_star"] = float(T_star)
        if offset_eclipse is not None:
            params_dict["offset_eclipse"] = float(offset_eclipse)

        if vmr_mode:
            for g, v in zip(gases[:-1], vmrs[:-1]):
                params_dict[f"log_{g}"] = float(np.log10(max(v, 1e-99)))
        else:
            params_dict["logZ"] = float(logZ)
            params_dict["CO_ratio"] = float(CO_ratio)
            params_dict["log_CH4_mult"] = float(np.log10(CH4_mult))

        binned, info = fn(params_dict)
        binned = np.asarray(binned, dtype=np.float64)
        wavelengths = self._bin_centres()

        if not full_output:
            return wavelengths, binned, None

        # info["unbinned_wavelengths"] is data["lambda_grid"], which may be a
        # further-trimmed view of atm.lambda_grid — keep them consistent.
        info_dict = _to_numpy(info)
        return wavelengths, binned, info_dict
