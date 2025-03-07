from . import _cupy_numpy as xp
import numpy as np
expn=xp.scipy.special.expn
import matplotlib.pyplot as plt
import scipy.special
gaussian_filter=scipy.ndimage.gaussian_filter

from astropy.io import ascii
import pandas as pd
from pkg_resources import resource_filename

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from ._atmosphere_solver import AtmosphereSolver
from ._interpolator_3D import interp1d, regular_grid_interp

class FluxCalculator:
    def __init__(self, include_condensation=True, method="xsec", include_opacities=["C2", "C3", "12C12C13C", "12C13C12C", "CN", "CO", "CP", "CS", "HCN", "NS", "OCS", "C2H2", "C2H4", "H2O", "toluene", "CH", "HeHp", "C2H"], downsample=1):
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
        
        
    def _exp3(self, x):
        shape = x.shape
        result = xp.interp(x.flatten(), self.tau_cache, self.exp3_cache,
                           left=0.5, right=0
                           )
        return result.reshape(shape)
        
    def change_wavelength_bins(self, bins):        
        '''Same functionality as :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)


    def _get_binned_fluxes(self, fluxes, disperser="g235H", n_gauss=10):
        if self.atm.wavelength_bins is None:
            return self.atm.lambda_grid, fluxes, self.atm.lambda_grid, fluxes

        fluxes = xp.cpu(fluxes)
        wavelengths = xp.cpu(self.atm.lambda_grid)
        if disperser == "prism":
            lsf_waves, lsf_res = np.loadtxt(resource_filename(__name__, "data/prism_resolution.txt"), unpack=True, delimiter=",")
        elif disperser == "g235H":
            lsf_waves, lsf_res = np.loadtxt(resource_filename(__name__, "data/g235H_resolution.txt"), unpack=True, delimiter=",")
        else:
            assert(False)
        lsf_waves *= 1e-6
        binned_fluxes = []
        wavelength_bins = xp.cpu(self.atm.wavelength_bins)
        
        for i in range(len(wavelength_bins)):
            wave = wavelength_bins[i].mean().item()
            res = np.interp(wave, lsf_waves, lsf_res)
            dw = wave / res
            cond = (wavelengths > wave - 5*dw) & (wavelengths < wave + 5 * dw)
            sigma = 15000 / res / 2.355
            smoothed_spec = gaussian_filter(fluxes[cond], sigma)
            binned_fluxes.append(np.interp(wave, wavelengths[cond], smoothed_spec))

        #plt.plot(wavelengths, fluxes)
        #plt.plot(wavelength_bins.mean(axis=1), binned_fluxes)
        #plt.show()
        #import pdb
        #pdb.set_trace()
        
        return self.atm.lambda_grid, fluxes, xp.array(wavelength_bins.mean(axis=1)), xp.array(binned_fluxes)
        
    def _get_photosphere_radii(self, taus, radii):
        intermediate_radii = 0.5 * (radii[0:-1] + radii[1:])
        result = radii[xp.argmin(xp.absolute(xp.log(taus)), axis=1)]
        return result

    def compute_fluxes(self, R_over_D, t_p_profile, planet_mass,
                       planet_radius, logZ=0, CO_ratio=0.53, CH4_mult=1,
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
                       surface_type=None, semimajor_axis=None, surface_temp=None, surface_pressure=xp.inf,
                       disperser="g235H"
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
            planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, CH4_mult, gases, vmrs, add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, bot_pressure, custom_abundances,
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
        '''print(padded_taus.shape)
        plt.imshow(np.log10(padded_taus.get()), aspect='auto')
        plt.colorbar()
        plt.figure()
        plt.plot(padded_taus[5000].get())
        plt.title("taus")
        plt.show()'''
        
        integrand = planck_function * xp.diff(self._exp3(padded_taus), axis=1)
        surf_fluxes = -2 * xp.pi * xp.sum(integrand, axis=1)
                
        if not xp.isinf(cloudtop_pressure) and cloudtop_pressure < surface_pressure:
            max_taus = taus.max(axis=1)
            fluxes_from_cloud = xp.pi * planck_function[:, -1] * (max_taus**2 * expn(1, max_taus) - max_taus * xp.exp(-max_taus) + xp.exp(-max_taus))
            surf_fluxes += fluxes_from_cloud

        fluxes = surf_fluxes * R_over_D**2 #from surface to Earth
        #fluxes = gaussian_filter(fluxes, 254) #R=15k to 150
        #self.smooth_to_prism_resolution(fluxes)
        unbinned_wavelengths, unbinned_fluxes, binned_wavelengths, binned_fluxes = self._get_binned_fluxes(fluxes, disperser)

        if full_output:
            atm_info["unbinned_fluxes"] = fluxes
            atm_info["unbinned_wavelengths"] = unbinned_wavelengths
            atm_info["taus"] = taus
            atm_info["contrib"] = -integrand / surf_fluxes[:, xp.newaxis]
            
            for key in atm_info:
                if type(atm_info[key]) == dict:
                    for subkey in atm_info[key]:
                        atm_info[key][subkey] = xp.cpu(atm_info[key][subkey])
                else:
                    atm_info[key] = xp.cpu(atm_info[key])
                
            return xp.cpu(binned_wavelengths), xp.cpu(binned_fluxes), atm_info

        return xp.cpu(binned_wavelengths), xp.cpu(binned_fluxes), None
