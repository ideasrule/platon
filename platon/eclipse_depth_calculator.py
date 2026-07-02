from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import ascii

from . import _forward_model as fm
from ._forward_model import ForwardConfig
from .constants import h, c, k_B, R_jup, M_jup, R_sun
from .errors import AtmosphereError
from ._atmosphere_solver import AtmosphereSolver
from .transit_depth_calculator import _prepare_forward_inputs, _atm_info_dict


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
            "xsec" for opacity sampling (correlated-k is no longer supported)
        '''
        self.atm = AtmosphereSolver(include_condensation, method=method,
                                    include_opacities=include_opacities,
                                    downsample=downsample)
        self.surface_library = surface_library

        if surface_library not in ["HES2012", "Paragas"]:
            raise ValueError(
                "The only surface libraries available are HES2012 and Paragas")

        basedir = Path(__file__).resolve().parent
        self.hemi_refls = pd.read_csv(
            basedir / f"data/{surface_library}/hemi_refls.csv")
        self.crust_emission_flux = ascii.read(
            basedir / f"data/{surface_library}/Crust_EmissionFlux.dat",
            delimiter="\t")
        if surface_library == "HES2012":
            self.redist_factors = {
                'Metal-rich': 0.6052, 'Ultramafic': 0.5532,
                'Feldspathic': 0.5414, 'Basaltic': 0.6004,
                'Granitoid': 0.5290, 'Clay': 0.5, 'Ice-rich silicate': 0.5,
                'Fe-oxidized': 0.5978}
        else:
            df = pd.read_csv(
                basedir / f"data/{surface_library}/f_relation_new_samples.csv")
            self.redist_factors = {col: df[col][0] for col in df.columns}
        self._surface_cache = {}

    def _get_surface_arrays(self, surface_type):
        """Reflectance interpolated onto the current and full wavelength
        grids, plus the crust emission flux lookup table."""
        key = (surface_type, self.atm.N_lambda)
        if key not in self._surface_cache:
            wl = np.asarray(self.hemi_refls["Wavelength"], dtype=np.float64)
            rh = np.asarray(self.hemi_refls[surface_type], dtype=np.float64)
            rh_binned = np.interp(self.atm.lambda_grid, wl, rh).astype(np.float32)
            rh_orig = np.interp(self.atm.orig_lambda_grid, wl, rh).astype(np.float32)
            crust_flux = np.asarray(
                self.crust_emission_flux[surface_type].data, np.float32)
            crust_T = np.asarray(
                self.crust_emission_flux["Temperature [K]"].data, np.float32)
            self._surface_cache[key] = (rh_binned, rh_orig, crust_flux, crust_T)
        return self._surface_cache[key]

    def calc_surface_temp(self, surface_type, stellar_fluxes_orig, a_over_Rs):
        """Host computation of the surface equilibrium temperature."""
        wl = np.asarray(self.hemi_refls["Wavelength"], dtype=np.float64)
        rh = np.asarray(self.hemi_refls[surface_type], dtype=np.float64)
        interp_rh = np.interp(self.atm.orig_lambda_grid, wl, rh)
        irrad = self.redist_factors[surface_type] * np.trapezoid(
            (1 - interp_rh) * np.asarray(stellar_fluxes_orig) / a_over_Rs**2,
            self.atm.orig_lambda_grid)
        flux_table = self.crust_emission_flux[surface_type].data
        if irrad < flux_table[0] or irrad > flux_table[-1]:
            raise ValueError("Cannot compute surface temperature because "
                             "irradiation is out of range of the data files")
        temperature = np.interp(
            irrad, np.asarray(flux_table),
            np.asarray(self.crust_emission_flux["Temperature [K]"].data))
        return temperature

    def calc_surface_flux(self, surface_type, stellar_fluxes, a_over_Rs,
                          temperature):
        wl = np.asarray(self.hemi_refls["Wavelength"], dtype=np.float64)
        rh = np.asarray(self.hemi_refls[surface_type], dtype=np.float64)
        hemi_reflectance = np.interp(self.atm.lambda_grid, wl, rh)
        directional_emissivity = 1 - hemi_reflectance
        lam = self.atm.lambda_grid
        emitted_fluxes = directional_emissivity * np.pi * 2 * h * c**2 / \
            lam**5 / np.expm1(h * c / (lam * k_B * temperature))
        reflected_fluxes = np.asarray(stellar_fluxes) / a_over_Rs**2 * \
            hemi_reflectance
        return emitted_fluxes + reflected_fluxes

    def change_wavelength_bins(self, bins):
        '''Same functionality as :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)
        self._surface_cache = {}

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
                       stellar_blackbody=False, full_output=False,
                       zero_opacities=[], surface_type=None,
                       semimajor_axis=None, surface_temp=None,
                       surface_pressure=np.inf):
        '''Most parameters are explained in :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`

        Parameters
        ----------
        t_p_profile : Profile
            A Profile object from TP_profile
        '''
        T_profile = np.asarray(t_p_profile.temperatures, dtype=np.float64)
        P_profile = np.asarray(t_p_profile.pressures, dtype=np.float64)
        bot_pressure = min(cloudtop_pressure, surface_pressure)

        has_surface = surface_pressure < cloudtop_pressure
        a_over_Rs = 0.0
        redist = 0.0
        if has_surface:
            if surface_type is None:
                raise ValueError(
                    "Must specify surface_type when surface_pressure is set")
            a_over_Rs = semimajor_axis / star_radius
            redist = self.redist_factors[surface_type]

        config_kwargs, inputs, host = _prepare_forward_inputs(
            self.atm, star_radius, planet_mass, planet_radius,
            P_profile, T_profile, logZ, CO_ratio, CH4_mult, gases, vmrs,
            add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
            number_density, part_size, part_size_std, P_quench,
            1e-99, 1e-99, zero_opacities, stellar_blackbody,
            bot_pressure=bot_pressure, n_t_rows=self.atm.N_T,
            surface_pressure=surface_pressure, a_over_Rs=a_over_Rs,
            surface_temp=surface_temp, redist=redist)

        cfg = ForwardConfig(has_surface=has_surface,
                            surface_temp_given=surface_temp is not None,
                            **config_kwargs)

        if has_surface:
            rh_binned, rh_orig, crust_flux, crust_T = \
                self._get_surface_arrays(surface_type)
            inputs = inputs._replace(rh_binned=rh_binned, rh_orig=rh_orig,
                                     crust_flux=crust_flux, crust_T=crust_T)

        out = fm.eclipse_core(cfg, self.atm.device_data(), inputs)

        if bool(out.atm.unbound):
            raise AtmosphereError("Atmosphere unbound: height > hill radius")

        if has_surface and surface_temp is None:
            irrad = float(out.irrad)
            flux_table = self.crust_emission_flux[surface_type].data
            if irrad < flux_table[0] or irrad > flux_table[-1]:
                raise ValueError(
                    "Cannot compute surface temperature because irradiation "
                    "is out of range of the data files")

        binned_depths = np.array(out.binned_depths, dtype=np.float64)
        if self.atm.wavelength_bins is None:
            binned_wavelengths = np.array(self.atm.lambda_grid)
        else:
            binned_wavelengths = np.array(
                self.atm._bin_info["eclipse_wavelengths"])

        if not full_output:
            return binned_wavelengths, binned_depths, None

        n = host["n_above"]
        atm_info = _atm_info_dict(self.atm, out, host)
        fluxes = np.array(out.fluxes, dtype=np.float64)
        integrand = np.array(out.integrand)[:, :n - 1]
        atm_info["surface_temp"] = float(out.surface_temp) if has_surface \
            else surface_temp
        atm_info["stellar_spectrum"] = np.array(out.stellar_spectrum,
                                                dtype=np.float64)
        atm_info["planet_spectrum"] = fluxes
        atm_info["unbinned_wavelengths"] = np.array(self.atm.lambda_grid)
        atm_info["unbinned_eclipse_depths"] = np.array(out.depths,
                                                       dtype=np.float64)
        atm_info["taus"] = np.array(out.taus)[:, :n - 1]
        atm_info["contrib"] = -integrand / fluxes[:, np.newaxis]
        atm_info["photosphere_radii"] = np.array(out.photosphere_radii)

        return binned_wavelengths, binned_depths, atm_info
