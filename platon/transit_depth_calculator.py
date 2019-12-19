import os
import sys

from pkg_resources import resource_filename
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate
import scipy.ndimage
from scipy.stats import lognorm

from . import _hydrostatic_solver
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from . import _interpolator_3D
from ._tau_calculator import get_line_of_sight_tau
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from ._get_data import get_data
from ._mie_cache import MieCache
from .errors import AtmosphereError
from ._atmosphere_solver import AtmosphereSolver


class TransitDepthCalculator:
    def __init__(self, include_condensation=True, num_profile_heights=250,
                 ref_pressure=1e5, method='xsec'):
        self.atm = AtmosphereSolver(include_condensation, num_profile_heights,
                               ref_pressure, method)               

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

        Raises
        ------
        NotImplementedError
            Raised when `change_wavelength_bins` is called more than once,
            which is not supported.
        """
        self.atm.change_wavelength_bins(bins)
        

    def _get_binned_corrected_depths(self, depths, T_star, T_spot,
                                  spot_cov_frac, n_gauss=10, blackbody=True):
        unbinned_lambdas = self.atm.lambda_grid
        stellar_spectrum, correction_factors = self.atm.get_stellar_spectrum(
            unbinned_lambdas, T_star, T_spot, spot_cov_frac, blackbody)
        
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

            for chunk in range(num_binned):
                start = chunk * n_gauss
                end = (chunk + 1 ) * n_gauss
                intermediate_depths[chunk] = np.sum(depths[start : end] * weights)

            intermediate_lambdas = unbinned_lambdas[::n_gauss]
            intermediate_stellar_spectrum = stellar_spectrum[::n_gauss]
            intermediate_correction_factors = correction_factors[::n_gauss]
            
        elif self.atm.method == "xsec":
            intermediate_lambdas = unbinned_lambdas
            intermediate_depths = depths
            intermediate_stellar_spectrum = stellar_spectrum
            intermediate_correction_factors = correction_factors
        else:
            assert(False)                  
                
        if self.atm.wavelength_bins is None:
            return intermediate_lambdas,\
                intermediate_depths * intermediate_correction_factors,\
                intermediate_stellar_spectrum
        
        binned_wavelengths = []
        binned_depths = []
        binned_stellar_spectrum = []
        
        for (start, end) in self.atm.wavelength_bins:
            cond = np.logical_and(
                intermediate_lambdas >= start,
                intermediate_lambdas < end)
            binned_wavelengths.append(np.mean(intermediate_lambdas[cond]))
            binned_depth = np.average(intermediate_depths[cond] * intermediate_correction_factors[cond],
                                      weights=intermediate_stellar_spectrum[cond])
            binned_depths.append(binned_depth)
            binned_stellar_spectrum.append(np.median(intermediate_stellar_spectrum[cond]))

        return np.array(binned_wavelengths), np.array(binned_depths), np.array(binned_stellar_spectrum)

    def _validate_params(self, T, logZ, CO_ratio, cloudtop_pressure):
        T_profile = np.ones(self.atm.num_profile_heights) * T
        self.atm._validate_params(T, logZ, CO_ratio, cloudtop_pressure)
        
    
    def compute_depths(self, star_radius, planet_mass, planet_radius,
                       temperature, logZ=0, CO_ratio=0.53,
                       add_gas_absorption=True,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       full_output=False, min_abundance=1e-99, min_cross_sec=1e-99, stellar_blackbody=False):
        '''
        Computes transit depths at a range of wavelengths, assuming an
        isothermal atmosphere.  To choose bins, call change_wavelength_bins().

        Parameters
        ----------
        star_radius : float
            Radius of the star
        planet_mass : float
            Mass of the planet, in kg
        planet_radius : float
            Radius of the planet at 100,000 Pa. Must be in metres.
        temperature : float
            Temperature of the isothermal atmosphere, in Kelvin
        logZ : float
            Base-10 logarithm of the metallicity, in solar units
        CO_ratio : float, optional
            C/O atomic ratio in the atmosphere.  The solar value is 0.53.
        add_gas_absorption: float, optional
            Whether gas absorption is accounted for
        add_scattering : bool, optional
            whether Rayleigh scattering is taken into account
        scattering_factor : float, optional
            if `add_scattering` is True, make scattering this many
            times as strong. If `scattering_slope` is 4, corresponding to
            Rayleigh scattering, the absorption coefficients are simply
            multiplied by `scattering_factor`. If slope is not 4,
            `scattering_factor` is defined such that the absorption coefficient
            is that many times as strong as Rayleigh scattering at
            `scattering_ref_wavelength`.
        scattering_slope : float, optional
            Wavelength dependence of scattering, with 4 being Rayleigh.
        scattering_ref_wavelength : float, optional
            Scattering is `scattering_factor` as strong as Rayleigh at this
            wavelength, expressed in metres.
        add_collisional_absorption : float, optional
            Whether collisionally induced absorption is taken into account
        cloudtop_pressure : float, optional
            Pressure level (in Pa) below which light cannot penetrate.
            Use np.inf for a cloudless atmosphere.
        custom_abundances : str or dict of np.ndarray, optional
            If specified, overrides `logZ` and `CO_ratio`.  Can specify a
            filename, in which case the abundances are read from a file in the
            format of the EOS/ files.  These are identical to ExoTransmit's
            EOS files.  It is also possible, though highly discouraged, to
            specify a dictionary mapping species names to numpy arrays, so that
            custom_abundances['Na'][3,4] would mean the fractional number
            abundance of Na at a temperature of self.T_grid[3] and pressure of
            self.P_grid[4].
        custom_T_profile : array-like, optional
            If specified and custom_P_profile is also specified, divides the
            atmosphere into user-specified P/T points, instead of assuming an
            isothermal atmosphere with T = `temperature`.
        custom_P_profile : array-like, optional
            Must be specified along with `custom_T_profile` to use a custom
            P/T profile.  Pressures must be in Pa.
        T_star : float, optional
            Effective temperature of the star.  If you specify this and
            use wavelength binning, the wavelength binning becomes
            more accurate.
        T_spot : float, optional
            Effective temperature of the star spots. This can be used to make
            wavelength dependent correction to the observed transit depths.
        spot_cov_frac : float, optional
            The spot covering fraction of the star by area. This can be used to
            make wavelength dependent correction to the transit depths.
        ri : complex, optional
            Complex refractive index n - ik (where k > 0) of the particles
            responsible for Mie scattering.  If provided, Mie scattering will
            be computed.  In that case, scattering_factor and scattering_slope
            must be set to 1 and 4 (the default values) respectively.
        frac_scale_height : float, optional
            The number density of Mie scattering particles is proportional to
            P^(1/frac_scale_height).  This is similar to, but a bit different
            from, saying that the scale height of the particles is
            frac_scale_height times that of the gas.
        number_density: float, optional
            The number density (in m^-3) of Mie scattering particles
        part_size : float, optional
            The mean radius of Mie scattering particles.  The distribution is
            assumed to be log-normal, with a standard deviation of part_size_std
        part_size_std : float, optional
            The geometric standard deviation of particle radii. We recommend
            leaving this at the default value of 0.5.
        full_output : bool, optional
            If True, returns info_dict as a third return value.

        Raises
        ------
        ValueError
            Raised when invalid parameters are passed to the method

        Returns
        -------
        wavelengths : array of float
            Central wavelengths, in metres
        transit_depths : array of float
            Transit depths at `wavelengths`
        info_dict : dict
            Returned if full_output is True, containing intermediate quantities
            calculated by the method.  These are: absorption_coeff_atm, tau_los,
            stellar_spectrum, radii, P_profile, T_profile, mu_profile,
            atm_abundances, unbinned_depths, unbinned_wavelengths
       '''
        if custom_P_profile is not None:
            if custom_T_profile is None or len(
                    custom_P_profile) != len(custom_T_profile):
                raise ValueError("Must specify both custom_T_profile and "
                                 "custom_P_profile, and the two must have the"
                                 " same length")
            if temperature is not None:
                raise ValueError(
                    "Cannot specify both temperature and custom T profile")
            
            P_profile = custom_P_profile
            T_profile = custom_T_profile
        else:
            P_profile = np.logspace(
                np.log10(self.atm.P_grid[0]),
                np.log10(self.atm.P_grid[-1]),
                self.atm.num_profile_heights)
            T_profile = np.ones(len(P_profile)) * temperature

        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, add_gas_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
            number_density, part_size, part_size_std, P_quench)

        radii = atm_info["radii"]
        dr = atm_info["dr"]
        tau_los = get_line_of_sight_tau(atm_info["absorption_coeff_atm"],
                                        radii)
        absorption_fraction = 1 - np.exp(-tau_los)

        transit_depths = (np.min(radii) / star_radius)**2 \
            + 2 / star_radius**2 * absorption_fraction.dot(radii[1:] * dr)
        binned_wavelengths, binned_depths, binned_stellar_spectrum = self._get_binned_corrected_depths(transit_depths, T_star, T_spot, spot_cov_frac, stellar_blackbody)
        
        if full_output:
            atm_info["tau_los"] = tau_los
            atm_info["binned_stellar_spectrum"] = binned_stellar_spectrum
            atm_info["unbinned_depths"] = transit_depths
            
            return binned_wavelengths, binned_depths, atm_info

        return binned_wavelengths, binned_depths
        
        
