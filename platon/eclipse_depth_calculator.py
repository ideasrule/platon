import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from .transit_depth_calculator import TransitDepthCalculator

try:
    import gnumpy as gnp
except ImportError:
    gnp = None
    print("Failed to import gnumpy; not using GPU")


class EclipseDepthCalculator:
    def __init__(self):
        self.transit_calculator = TransitDepthCalculator()
        self.wavelength_bins = None
        self.d_ln_lambda = np.median(np.diff(np.log(self.transit_calculator.lambda_grid)))

    def change_wavelength_bins(self, bins):
        self.transit_calculator.change_wavelength_bins(bins)
        self.wavelength_bins = bins

    def _get_binned_depths(self, depths, stellar_spectrum):
        if self.wavelength_bins is None:
            return self.transit_calculator.lambda_grid, depths
        
        binned_wavelengths = []
        binned_depths = []
        for (start, end) in self.wavelength_bins:
            cond = np.logical_and(
                self.transit_calculator.lambda_grid >= start,
                self.transit_calculator.lambda_grid < end)
            binned_wavelengths.append(np.mean(self.transit_calculator.lambda_grid[cond]))
            binned_depth = np.average(depths[cond], weights=stellar_spectrum[cond])
            binned_depths.append(binned_depth)
            
        return np.array(binned_wavelengths), np.array(binned_depths)
            
    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, T_star, logZ=0, CO_ratio=0.53,
                       add_gas_absorption=True,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_spot=None, spot_cov_frac=None,
                       ri = None, frac_scale_height=1,number_density=0,
                       part_size=1e-6, num_mu=50, min_mu=1e-2, max_mu=1,
                       full_output=False):

        T_profile = t_p_profile.temperatures
        P_profile = t_p_profile.pressures
        wavelengths, transit_depths, info_dict = self.transit_calculator.compute_depths(
            star_radius, planet_mass, planet_radius, None, logZ, CO_ratio,
            add_gas_absorption,
            add_scattering, scattering_factor,
            scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_profile,
            P_profile,
            T_star, T_spot, spot_cov_frac,
            ri, frac_scale_height, number_density, part_size, full_output=True)

        absorption_coeff = info_dict["absorption_coeff_atm"]
        intermediate_coeff = 0.5 * (absorption_coeff[0:-1] + absorption_coeff[1:])
        intermediate_T = 0.5 * (info_dict["T_profile"][0:-1] + info_dict["T_profile"][1:])
        dr = -np.diff(info_dict["radii"])
        d_taus = intermediate_coeff.T * dr
        taus = np.cumsum(d_taus, axis=1)

        mu_grid = np.logspace(np.log10(min_mu), np.log10(max_mu), num_mu)
        d_mus = np.diff(mu_grid)
        d_mus = np.append(d_mus[0], d_mus)
        lambda_grid = self.transit_calculator.lambda_grid
        
        if gnp is not None:
            reshaped_lambda_grid = gnp.garray(lambda_grid.reshape((-1, 1)))
            planck_function = 2*h*c**2/reshaped_lambda_grid**5/(gnp.exp(h*c/reshaped_lambda_grid/k_B/gnp.garray(intermediate_T)) - 1)
            reshaped_taus = gnp.garray(taus[:,:,np.newaxis])
            reshaped_planck = planck_function[:,:,np.newaxis]
            reshaped_d_taus = gnp.garray(d_taus[:,:,np.newaxis])
            integrands = gnp.exp(-reshaped_taus/gnp.garray(mu_grid)) * reshaped_planck * reshaped_d_taus
            fluxes = integrands.sum(axis=1).asarray()
        else:
            reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
            planck_function = ne.evaluate("2*h*c**2/reshaped_lambda_grid**5/(exp(h*c/reshaped_lambda_grid/k_B/intermediate_T) - 1)")
            reshaped_taus = taus[:,:,np.newaxis]
            reshaped_planck = planck_function[:,:,np.newaxis]
            reshaped_d_taus = d_taus[:,:,np.newaxis]
            reshaped_d_mus = d_mus[np.newaxis, np.newaxis, :]
            integrands = ne.evaluate("exp(-reshaped_taus/mu_grid) * reshaped_planck * reshaped_d_taus")
            fluxes = integrands.sum(axis=1)

        fluxes = 2 * np.pi * np.sum(fluxes * d_mus, axis=1)
        stellar_photon_fluxes = info_dict["stellar_spectrum"]
        d_lambda = self.d_ln_lambda * lambda_grid
        photon_fluxes = fluxes * d_lambda / (h * c / lambda_grid)

        eclipse_depths = photon_fluxes / stellar_photon_fluxes * (planet_radius/star_radius)**2

        binned_wavelengths, binned_depths = self._get_binned_depths(eclipse_depths, stellar_photon_fluxes)

        if full_output:
            output_dict = dict(info_dict)
            output_dict["planet_spectrum"] = fluxes
            output_dict["unbinned_eclipse_depths"] = eclipse_depths
            return binned_wavelengths, binned_depths, output_dict

        return binned_wavelengths, binned_depths
            


