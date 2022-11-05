import cupy as np
import matplotlib.pyplot as plt
from cupyx.scipy.special import expn
import time

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from ._atmosphere_solver import AtmosphereSolver

class EclipseDepthCalculator:
    def __init__(self, include_condensation=True, method="xsec"):
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
        self.atm = AtmosphereSolver(include_condensation=include_condensation, method=method)


        
    def change_wavelength_bins(self, bins):        
        '''Same functionality as :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)


    def _get_binned_depths(self, depths, stellar_spectrum, n_gauss=10):
        #Step 1: do a first binning if using k-coeffs; first binning is a
        #no-op otherwise
        if self.atm.method == "ktables":
            #Do a first binning based on ktables
            points, weights = scipy.special.roots_legendre(n_gauss)
            percentiles = 100 * (points + 1) / 2
            weights /= 2
            assert(len(depths) % n_gauss == 0)
            num_binned = int(len(depths) / n_gauss)
            intermediate_lambdas = np.zeros(num_binned)
            intermediate_depths = np.zeros(num_binned)
            intermediate_stellar_spectrum = np.zeros(num_binned)

            for chunk in range(num_binned):
                start = chunk * n_gauss
                end = (chunk + 1 ) * n_gauss
                intermediate_lambdas[chunk] = np.median(self.atm.lambda_grid[start : end])
                intermediate_depths[chunk] = np.sum(depths[start : end] * weights)
                intermediate_stellar_spectrum[chunk] = np.median(stellar_spectrum[start : end])
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
        for (start, end) in self.atm.wavelength_bins:
            cond = np.logical_and(
                intermediate_lambdas >= start,
                intermediate_lambdas < end)
            binned_wavelengths.append(np.mean(intermediate_lambdas[cond]))
            binned_depth = np.average(intermediate_depths[cond],
                                      weights=intermediate_stellar_spectrum[cond])
            binned_depths.append(binned_depth)
            
        return intermediate_lambdas, intermediate_depths, np.array(binned_wavelengths), np.array(binned_depths)

    def _get_photosphere_radii(self, taus, radii):
        intermediate_radii = 0.5 * (radii[0:-1] + radii[1:])
        photosphere_radii = np.array([np.interp(np.array([1]), t, intermediate_radii)[0] for t in taus])
        return photosphere_radii

    #@profile
    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, T_star, logZ=0, CO_ratio=0.53,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_spot=None, spot_cov_frac=None,
                       ri = None, frac_scale_height=1,number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       stellar_blackbody=False,
                       full_output=False):
        '''Most parameters are explained in :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`

        Parameters
        ----------
        t_p_profile : Profile
            A Profile object from TP_profile
        '''
        T_profile = t_p_profile.temperatures
        P_profile = t_p_profile.pressures
        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac,
            ri, frac_scale_height, number_density, part_size, part_size_std,
            P_quench)

        assert(np.max(atm_info["P_profile"]) <= cloudtop_pressure)
        absorption_coeff = atm_info["absorption_coeff_atm"]
        intermediate_coeff = 0.5 * (absorption_coeff[0:-1] + absorption_coeff[1:])
        intermediate_T = 0.5 * (atm_info["T_profile"][0:-1] + atm_info["T_profile"][1:])
        dr = atm_info["dr"]
        d_taus = intermediate_coeff.T * dr
        taus = np.cumsum(d_taus, axis=1)

        lambda_grid = self.atm.lambda_grid

        reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
        planck_function = 2*h*c**2/reshaped_lambda_grid**5/(np.exp(h*c/reshaped_lambda_grid/k_B/intermediate_T) - 1)

        #padded_taus: ensures 1st layer has 0 optical depth
        padded_taus = np.zeros((taus.shape[0], taus.shape[1] + 1))
        padded_taus[:, 1:] = taus
        integrand = planck_function * np.diff(expn(3, padded_taus), axis=1)
        fluxes = -2 * np.pi * np.sum(integrand, axis=1)
                
        if not np.isinf(cloudtop_pressure):
            max_taus = np.max(taus, axis=1)
            fluxes_from_cloud = -np.pi * planck_function[:, -1] * (max_taus**2 * -expn(max_taus) + max_taus * np.exp(-max_taus) - np.exp(-max_taus))
            fluxes += fluxes_from_cloud

        stellar_photon_fluxes, _ = self.atm.get_stellar_spectrum(
            lambda_grid, T_star, T_spot, spot_cov_frac, False)
        
        d_lambda = self.atm.d_ln_lambda * lambda_grid
        photon_fluxes = fluxes * d_lambda / (h * c / lambda_grid)

        photosphere_radii = planet_radius #self._get_photosphere_radii(taus, atm_info["radii"])
        eclipse_depths = photon_fluxes / stellar_photon_fluxes * (photosphere_radii/star_radius)**2

        #For correlated k, eclipse_depths has n_gauss points per wavelength, while unbinned_depths has 1 point per wavelength
        unbinned_wavelengths, unbinned_depths, binned_wavelengths, binned_depths = self._get_binned_depths(eclipse_depths, stellar_photon_fluxes)

        if full_output:
            atm_info["stellar_spectrum"] = stellar_photon_fluxes
            atm_info["planet_spectrum"] = fluxes
            atm_info["unbinned_wavelengths"] = unbinned_wavelengths
            atm_info["unbinned_eclipse_depths"] = unbinned_depths
            atm_info["taus"] = taus
            atm_info["contrib"] = -integrand / fluxes[:, np.newaxis]
            return binned_wavelengths, binned_depths, atm_info

        return binned_wavelengths, binned_depths
            


