from . import _cupy_numpy as xp
expn=xp.scipy.special.expn
import matplotlib.pyplot as plt
import scipy.special
from astropy.io import ascii
import pandas as pd
from pkg_resources import resource_filename

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from ._atmosphere_solver import AtmosphereSolver
from ._interpolator_3D import interp1d, regular_grid_interp

class EclipseDepthCalculator:
    def __init__(self, include_condensation=True, method="xsec", include_opacities=["CH4", "CO2", "CO", "H2O", "H2S", "HCN", "K", "Na", "NH3", "SO2", "TiO", "VO"], downsample=1,
                 surface_library="Paragas"):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        num_profile_heights : int
            The number of zones the atmosphere is divided into
        ref_pressure : float
            The planetary radius is defined as the radius at this pressure
        method : string
            "xsec" for opacity sampling, "ktables" for correlated k
        '''
        self.atm = AtmosphereSolver(include_condensation, method=method, include_opacities=include_opacities, downsample=downsample)
        self.tau_cache = xp.logspace(-6, 3, 1000)
        self.exp3_cache = expn(3, self.tau_cache)
        self.surface_library = surface_library
        
        if surface_library not in ["HES2012", "Paragas"]:
            raise ValueError("The only surface libraries available are HES2012 and Paragas")
        self.hemi_refls = pd.read_csv(resource_filename(__name__, f"data/{surface_library}/hemi_refls.csv"))
        self.crust_emission_flux = ascii.read(resource_filename(__name__, f"data/{surface_library}/Crust_EmissionFlux.dat"), delimiter="\t")
        if surface_library == "HES2012":
            self.redist_factors = {'Metal-rich': 0.6052, 'Ultramafic': 0.5532, 'Feldspathic': 0.5414,
                        'Basaltic': 0.6004, 'Granitoid': 0.5290, 'Clay': 0.5, 'Ice-rich silicate': 0.5,
                        'Fe-oxidized': 0.5978}
        else:
            df = pd.read_csv(resource_filename(__name__, f"data/{surface_library}/f_relation_new_samples.csv"))
            self.redist_factors = {col: df[col][0] for col in df.columns}
        
        
    def calc_surface_flux(self, surface_type, stellar_fluxes, stellar_fluxes_orig, Rp_over_Rs, a_over_Rs, temperature=None):
        if temperature is None:
            interp_rh = xp.interp(self.atm.orig_lambda_grid, xp.asarray(self.hemi_refls["Wavelength"]), xp.asarray(self.hemi_refls[surface_type]))
            irrad = self.redist_factors[surface_type] * xp.trapz(
                (1 - interp_rh) * stellar_fluxes_orig / a_over_Rs**2,
                                     self.atm.orig_lambda_grid)
            if irrad < self.crust_emission_flux[surface_type].data[0] or irrad > self.crust_emission_flux[surface_type].data[-1]:
                raise ValueError("Cannot compute surface temperature because irradiation is out of range of the data files")
            
            temperature = xp.interp(irrad, xp.array(self.crust_emission_flux[surface_type].data), xp.array(self.crust_emission_flux["Temperature [K]"]))

        hemi_reflectance = xp.interp(self.atm.lambda_grid, xp.array(self.hemi_refls["Wavelength"]), xp.array(self.hemi_refls[surface_type]))
        directional_emissivity = 1 - hemi_reflectance
        emitted_fluxes = directional_emissivity * xp.pi * 2 * h * c**2 / self.atm.lambda_grid**5 / xp.expm1(h*c/(self.atm.lambda_grid * k_B * temperature))
        reflected_fluxes = stellar_fluxes  / a_over_Rs**2 * hemi_reflectance
        fluxes = emitted_fluxes + reflected_fluxes
        return fluxes
        
    def _exp3(self, x):
        shape = x.shape
        result = xp.interp(x.flatten(), self.tau_cache, self.exp3_cache,
                           left=0.5, right=0
                           )
        return result.reshape(shape)
        
    def change_wavelength_bins(self, bins):        
        '''Same functionality as :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)


    def _get_binned_depths(self, depths, stellar_spectrum, n_gauss=10):
        #Step 1: do a first binning if using k-coeffs; first binning is a
        #no-op otherwise
        if self.atm.method == "ktables":
            #Do a first binning based on ktables
            points, weights = xp.array(scipy.special.roots_legendre(n_gauss))
            percentiles = 100 * (points + 1) / 2
            weights /= 2
            assert(len(depths) % n_gauss == 0)
            num_binned = int(len(depths) / n_gauss)
            intermediate_lambdas = xp.zeros(num_binned)
            intermediate_depths = xp.zeros(num_binned)
            intermediate_stellar_spectrum = xp.zeros(num_binned)

            for chunk in range(num_binned):
                start = chunk * n_gauss
                end = (chunk + 1 ) * n_gauss
                intermediate_lambdas[chunk] = xp.median(self.atm.lambda_grid[start : end])
                intermediate_depths[chunk] = xp.sum(depths[start : end] * weights)
                intermediate_stellar_spectrum[chunk] = xp.median(stellar_spectrum[start : end])
        elif self.atm.method == "xsec":
            intermediate_lambdas = self.atm.lambda_grid
            intermediate_depths = depths
            intermediate_stellar_spectrum = stellar_spectrum
        else:
            assert(False)

        
        if self.atm.wavelength_bins is None:
            return intermediate_lambdas, intermediate_depths, intermediate_lambdas, intermediate_depths
        
        binned_wavelengths = []
        binned_depths = []
        intermediate_stellar_photon_spectrum = intermediate_stellar_spectrum / (h * c / intermediate_lambdas)
        for (start, end) in self.atm.wavelength_bins:
            cond = xp.logical_and(
                intermediate_lambdas >= start,
                intermediate_lambdas < end)
            binned_wavelengths.append(xp.mean(intermediate_lambdas[cond]))
            binned_depth = xp.average(intermediate_depths[cond],
                                      weights=intermediate_stellar_photon_spectrum[cond])
            binned_depths.append(binned_depth)
            
        return intermediate_lambdas, intermediate_depths, xp.array(binned_wavelengths), xp.array(binned_depths)

    def _get_photosphere_radii(self, taus, radii, planet_radius):
        intermediate_radii = 0.5 * (radii[0:-1] + radii[1:])
        max_taus = xp.max(taus, axis=1)
        result = radii[xp.argmin(xp.absolute(xp.log(taus)), axis=1)]
        result[max_taus < 1] = planet_radius
        return result
              
    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, T_star, logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf, custom_abundances=None,
                       T_spot=None, spot_cov_frac=None,
                       ri = None, frac_scale_height=1,number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       stellar_blackbody=False,
                       full_output=False, zero_opacities=[],
                       surface_type=None, semimajor_axis=None, surface_temp=None, surface_pressure=xp.inf
                       ):
        '''Most parameters are explained in :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`

        Parameters
        ----------
        t_p_profile : Profile
            A Profile object from TP_profile
        '''
        T_profile = t_p_profile.temperatures
        P_profile = t_p_profile.pressures
        bot_pressure = min(cloudtop_pressure, surface_pressure)
        
        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, CH4_mult, gases, vmrs, add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, bot_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac,
            ri, frac_scale_height, number_density, part_size, part_size_std,
            P_quench, zero_opacities=zero_opacities)

        assert(atm_info["P_profile"].max() <= bot_pressure)
        absorption_coeff = atm_info["absorption_coeff_atm"]
        intermediate_coeff = 0.5 * (absorption_coeff[0:-1] + absorption_coeff[1:])
        intermediate_T = 0.5 * (atm_info["T_profile"][0:-1] + atm_info["T_profile"][1:])
        dr = atm_info["dr"]
        d_taus = intermediate_coeff.T * dr
        taus = xp.cumsum(d_taus, axis=1)

        lambda_grid = self.atm.lambda_grid

        reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
        planck_function = 2*h*c**2/reshaped_lambda_grid**5/(xp.exp(h*c/reshaped_lambda_grid/k_B/intermediate_T) - 1)

        #padded_taus: ensures 1st layer has 0 optical depth
        padded_taus = xp.zeros((taus.shape[0], taus.shape[1] + 1))
        padded_taus[:, 1:] = taus
        integrand = planck_function * xp.diff(self._exp3(padded_taus), axis=1)
        fluxes = -2 * xp.pi * xp.sum(integrand, axis=1)
                
        if not xp.isinf(cloudtop_pressure) and cloudtop_pressure < surface_pressure:
            max_taus = taus.max(axis=1)
            fluxes_from_cloud = xp.pi * planck_function[:, -1] * (max_taus**2 * expn(1, max_taus) - max_taus * xp.exp(-max_taus) + xp.exp(-max_taus))
            fluxes += fluxes_from_cloud

        stellar_fluxes, _ = self.atm.get_stellar_spectrum(
            lambda_grid, T_star, T_spot, spot_cov_frac, stellar_blackbody)
        
        if surface_pressure < cloudtop_pressure:
            stellar_fluxes_orig, _ = self.atm.get_stellar_spectrum(
                self.atm.orig_lambda_grid, T_star, T_spot, spot_cov_frac, stellar_blackbody)
            surface_flux = self.calc_surface_flux(surface_type, stellar_fluxes, stellar_fluxes_orig, planet_radius / star_radius, semimajor_axis / star_radius, surface_temp)
            max_taus = taus.max(axis=1)
            fluxes += surface_flux * (max_taus**2 * expn(1, max_taus) - max_taus * xp.exp(-max_taus) + xp.exp(-max_taus))
        
        photosphere_radii = self._get_photosphere_radii(taus, atm_info["radii"], planet_radius)
        eclipse_depths = fluxes / stellar_fluxes * (photosphere_radii/star_radius)**2
        #For correlated k, eclipse_depths has n_gauss points per wavelength, while unbinned_depths has 1 point per wavelength
        unbinned_wavelengths, unbinned_depths, binned_wavelengths, binned_depths = self._get_binned_depths(eclipse_depths, stellar_fluxes)

        if full_output:
            atm_info["stellar_spectrum"] = stellar_fluxes
            atm_info["planet_spectrum"] = fluxes
            atm_info["unbinned_wavelengths"] = unbinned_wavelengths
            atm_info["unbinned_eclipse_depths"] = unbinned_depths
            atm_info["taus"] = taus
            atm_info["contrib"] = -integrand / fluxes[:, xp.newaxis]
            
            for key in atm_info:
                if type(atm_info[key]) == dict:
                    for subkey in atm_info[key]:
                        atm_info[key][subkey] = xp.cpu(atm_info[key][subkey])
                else:
                    atm_info[key] = xp.cpu(atm_info[key])
                
            return xp.cpu(binned_wavelengths), xp.cpu(binned_depths), atm_info

        return xp.cpu(binned_wavelengths), xp.cpu(binned_depths), None
