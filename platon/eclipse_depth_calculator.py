import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from .transit_depth_calculator import TransitDepthCalculator

class EclipseDepthCalculator:
    def __init__(self):
        self.transit_calculator = TransitDepthCalculator()

    def compute_depths(self, alpha, T_0, star_radius, planet_mass,
                         planet_radius, T_star, logZ=0, CO_ratio=0.53,
                         add_scattering=True, scattering_factor=1,
                         scattering_slope=4, scattering_ref_wavelength=1e-6,
                         add_collisional_absorption=True,
                         cloudtop_pressure=np.inf, custom_abundances=None,
                         custom_T_profile=None, custom_P_profile=None,
                         T_spot=None, spot_cov_frac=None,
                         ri = None, frac_scale_height=1,number_density=0,
                         part_size = 10**-6, num_mu=100, min_mu=1e-3, max_mu=1):
        
        if custom_T_profile is not None and custom_P_profile is not None:
            T_profile = custom_T_profile
            P_profile = custom_P_profile
        else:
            assert(custom_T_profile is None)
            assert(custom_P_profile is None)
            P_profile = self.transit_calculator.P_grid
            T_profile = 1.0/alpha**2 * np.log(P_profile/np.min(P_profile))**2 + T_0

        wavelengths, transit_depths, info_dict = self.transit_calculator.compute_depths(
            star_radius, planet_mass, planet_radius, None, logZ, CO_ratio,
            add_scattering, scattering_factor,
            scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_profile,
            P_profile,
            T_star, T_spot, spot_cov_frac,
            ri, frac_scale_height, number_density, part_size, full_output=True)

        absorption_coeff = info_dict["absorption_coeff_atm"]
        intermediate_coeff = 0.5 * (absorption_coeff[:, 0:-1] + absorption_coeff[:, 1:])
        intermediate_T = 0.5 * (T_profile[0:-1] + T_profile[1:])
        dr = -np.diff(info_dict["radii"])
        d_taus = intermediate_coeff * dr
        taus = np.cumsum(d_taus, axis=1)

        mu_grid = np.linspace(min_mu, max_mu, num_mu)
        d_mu = (max_mu - min_mu)/(num_mu - 1)
        lambda_grid = self.transit_calculator.lambda_grid
        reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
        planck_function = ne.evaluate("2*h*c**2/reshaped_lambda_grid**5/exp(h*c/reshaped_lambda_grid/k_B/intermediate_T - 1)")

        reshaped_taus = taus[:,:,np.newaxis]
        reshaped_planck = planck_function[:,:,np.newaxis]
        reshaped_d_taus = d_taus[:,:,np.newaxis]
        integrands = ne.evaluate("exp(-reshaped_taus/mu_grid) * reshaped_planck * reshaped_d_taus")

        fluxes = 2 * np.pi * np.sum(integrands, axis=(1, 2)) * d_mu

        # Extra factor of pi/1e7 is due to bug in stellar spectrum; will be
        # removed soon
        stellar_photon_fluxes = info_dict["stellar_spectrum"] / 1e7 * np.pi
        
        d_lambda = np.diff(lambda_grid)
        d_lambda = np.append(d_lambda, d_lambda[-1])
        photon_fluxes = fluxes * d_lambda / (h * c / lambda_grid)
        eclipse_depths = photon_fluxes / stellar_photon_fluxes * transit_depths

        return lambda_grid, eclipse_depths
