from __future__ import print_function

import os
import sys

from pkg_resources import resource_filename
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate

from ._compatible_loader import load_dict_from_pickle
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from . import _interpolator_3D
from ._tau_calculator import get_line_of_sight_tau
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from ._get_data import get_data
from .errors import AtmosphereError

class TransitDepthCalculator:
    def __init__(self, include_condensation=True, min_P_profile=0.1, max_P_profile=1e5, num_profile_heights=400):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        min_P_profile : float
            For the radiative transfer calculation, the atmosphere is divided
            into zones.  This is the pressure at the topmost zone.
        max_P_profile: float
            The pressure at the bottommost zone of the atmosphere
        num_profile_heights : int
            The number of zones the atmosphere is divided into
        '''
        
        if not os.path.isdir(resource_filename(__name__, "data/")):
            get_data(resource_filename(__name__, "./"))
            
        self.stellar_spectra = load_dict_from_pickle(
            resource_filename(__name__, "data/stellar_spectra.pkl"))
        self.absorption_data, self.mass_data, self.polarizability_data = read_species_data(
            resource_filename(__name__, "data/Absorption"),
            resource_filename(__name__, "data/species_info"))
        self.collisional_absorption_data = load_dict_from_pickle(
            resource_filename(__name__, "data/collisional_absorption.pkl"))
        self.lambda_grid = np.load(
            resource_filename(__name__, "data/wavelengths.npy"))
        self.P_grid = np.load(
            resource_filename(__name__, "data/pressures.npy"))
        self.T_grid = np.load(
            resource_filename(__name__, "data/temperatures.npy"))

        self.N_lambda = len(self.lambda_grid)
        self.N_T = len(self.T_grid)
        self.N_P = len(self.P_grid)

        P_meshgrid, lambda_meshgrid, T_meshgrid = np.meshgrid(
            self.P_grid, self.lambda_grid, self.T_grid)
        self.P_meshgrid = P_meshgrid
        self.T_meshgrid = T_meshgrid

        self.wavelength_rebinned = False
        self.wavelength_bins = None

        self.abundance_getter = AbundanceGetter(include_condensation)

        self.min_P_profile = min_P_profile
        self.max_P_profile = max_P_profile
        self.num_profile_heights = num_profile_heights

        if min_P_profile <= self.P_grid[0] or min_P_profile >= self.P_grid[-1]:
            raise ValueError(
                "min_P_profile {} Pa out of bounds ({} to {} Pa)".format(
                    min_P_profile, self.P_grid[0], self.P_grid[-1]))
        
        if max_P_profile <= self.P_grid[0] or max_P_profile >= self.P_grid[-1]:
            raise ValueError(
                "max_P_profile {} Pa out of bounds ({} to {} Pa)".format(
                    max_P_profile, self.P_grid[0], self.P_grid[-1]))
        
        if min_P_profile >= max_P_profile:
            raise ValueError("min_P_profile must be less than max_P_profile")
                            

    def change_wavelength_bins(self, bins):
        """Specify wavelength bins, instead of using the full wavelength grid
        in self.lambda_grid.  This makes the code much faster, as
        `compute_depths` will only compute depths at wavelengths that fall
        within a bin.

        Parameters
        ----------
        bins : array_like, shape (N,2)
            Wavelength bins, where bins[i][0] is the start wavelength and
            bins[i][1] is the end wavelength for bin i.

        Raises
        ------
        NotImplementedError
            Raised when `change_wavelength_bins` is called more than once,
            which is not supported.
        """
        if self.wavelength_rebinned:
            raise NotImplementedError("Multiple re-binnings not yet supported")

        self.wavelength_rebinned = True
        self.wavelength_bins = bins

        cond = np.any([np.logical_and(self.lambda_grid > start, self.lambda_grid < end) for (start,end) in bins], axis=0)

        for key in self.absorption_data:
            self.absorption_data[key] = self.absorption_data[key][cond]

        for key in self.collisional_absorption_data:
            self.collisional_absorption_data[key] = self.collisional_absorption_data[key][cond]

        self.lambda_grid = self.lambda_grid[cond]
        self.N_lambda = len(self.lambda_grid)

        P_meshgrid, lambda_meshgrid, T_meshgrid = np.meshgrid(self.P_grid, self.lambda_grid, self.T_grid)
        self.P_meshgrid = P_meshgrid
        self.T_meshgrid = T_meshgrid
        
        for Teff in self.stellar_spectra:
            self.stellar_spectra[Teff] = self.stellar_spectra[Teff][cond]


    def _get_gas_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = np.zeros((self.N_lambda, np.sum(P_cond), np.sum(T_cond)))
        for species_name, species_abundance in abundances.items():
            assert(species_abundance.shape == (self.N_P, self.N_T))
            if species_name in self.absorption_data:
                absorption_coeff += self.absorption_data[species_name][:,P_cond,:][:,:,T_cond] * species_abundance[P_cond,:][:,T_cond]

        return absorption_coeff


    def _get_scattering_absorption(self, abundances, P_cond, T_cond,
                                   multiple=1, slope=4, ref_wavelength=1e-6):
        sum_polarizability_sqr = np.zeros((np.sum(P_cond), np.sum(T_cond)))

        for species_name in abundances:
            if species_name in self.polarizability_data:
                sum_polarizability_sqr += abundances[species_name][P_cond,:][:,T_cond] * self.polarizability_data[species_name]**2

        n = self.P_meshgrid[:,P_cond,:][:,:,T_cond]/(k_B*self.T_meshgrid[:,P_cond,:][:,:,T_cond])
        reshaped_lambda = self.lambda_grid.reshape((self.N_lambda, 1, 1))

        return multiple * (128.0/3 * np.pi**5) * n * sum_polarizability_sqr * ref_wavelength**(slope-4) / reshaped_lambda**slope


    def _get_collisional_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = np.zeros((self.N_lambda, np.sum(P_cond), np.sum(T_cond)))
        n = self.P_meshgrid[:,P_cond,:][:,:,T_cond]/(k_B * self.T_meshgrid[:,P_cond,:][:,:,T_cond])

        for s1, s2 in self.collisional_absorption_data:
            if s1 in abundances and s2 in abundances:
                n1 = (abundances[s1][P_cond, :][:,T_cond]*n)
                n2 = (abundances[s2][P_cond, :][:,T_cond]*n)
                abs_data = self.collisional_absorption_data[(s1,s2)].reshape((self.N_lambda, 1, self.N_T))[:,:,T_cond]
                absorption_coeff += abs_data * n1 * n2

        return absorption_coeff


    def _get_above_cloud_r_and_dr(self, P_profile, T_profile, abundances,
                                  planet_mass, planet_radius, star_radius,
                                  above_cloud_cond, T_star=None):
        assert(len(P_profile) == len(T_profile))
        # First, get atmospheric weight profile
        g = G * planet_mass/planet_radius**2        
        mu = np.zeros(len(P_profile))
        
        for species_name in abundances:
            interpolator = RectBivariateSpline(self.P_grid, self.T_grid, abundances[species_name], kx=1, ky=1)
            atm_abundances = interpolator.ev(P_profile, T_profile)
            mu += atm_abundances * self.mass_data[species_name]

        mu_interpolator = UnivariateSpline(P_profile, mu)
        T_interpolator = UnivariateSpline(P_profile, T_profile)
        
        # Ensure that the atmosphere is bound by making rough estimates of the
        # Hill radius and atmospheric height
        if T_star is None:
            T_star = Teff_sun
        R_hill = 0.5*star_radius*(T_star/T_profile[0])**2 * (planet_mass/(3*M_sun))**(1.0/3)
        scale_height = k_B*np.mean(T_profile)*planet_radius**2/(np.mean(mu) * AMU * G * planet_mass)
        atm_height_estimate = np.log(P_profile[-1]/P_profile[0]) * scale_height
        if atm_height_estimate > R_hill:
            raise AtmosphereError("Atmosphere unbound: height > hill radius")
        
        # Solve the hydrostatic equation
        def hydrostatic(y, P):
            r = y + planet_radius
            T_local = T_interpolator(P)
            local_mu = mu_interpolator(P)
            rho = local_mu*P*AMU / (k_B * T_local)
            dydP = -r**2/(G * planet_mass * rho)
            return dydP
        
        heights, infodict = integrate.odeint(hydrostatic, 0, P_profile[::-1],
                                             full_output=True)
        solver_errored = np.any(np.logical_and(infodict["mused"] != 1, infodict["mused"] != 2))
        if solver_errored:
            raise AtmosphereError("Hydrostatic solver failed")
        
        dr = np.diff(heights.flatten())
        dr = np.flipud(np.append(dr,k_B*T_profile[0]/(mu[0] * AMU * g)))
        radius_with_atm = planet_radius + np.sum(dr)

        radii = radius_with_atm - np.cumsum(dr)
        radii = np.append(radius_with_atm, radii[above_cloud_cond])
        return radii, dr[above_cloud_cond]

    def _get_abundances_array(self, logZ, CO_ratio, custom_abundances):
        if custom_abundances is None:
            return self.abundance_getter.get(logZ, CO_ratio)

        if logZ is not None or CO_ratio is not None:
            raise ValueError("Must set logZ=None and CO_ratio=None to use custom_abundances")

        if type(custom_abundances) is str:
            # Interpret as filename
            return AbundanceGetter.from_file(custom_abundances)

        if type(custom_abundances) is dict:
            for key, value in custom_abundances.items():
                if type(value) is not np.ndarray:
                    raise ValueError("custom_abundances must map species names to arrays")
                if value.shape != (self.N_P, self.N_T):
                    raise ValueError("custom_abundances has array of invalid size")
            return custom_abundances

        raise ValueError("Unrecognized format for custom_abundances")


    def _get_binned_depths(self, depths, T_star):
        if self.wavelength_bins is None:
            return self.lambda_grid, depths

        temperatures = list(self.stellar_spectra.keys())
        if T_star is None:
            stellar_spectrum = np.ones(len(self.lambda_grid))
        elif T_star >= np.min(temperatures) and T_star <= np.max(temperatures):
            interpolator = scipy.interpolate.interp1d(
                temperatures, list(self.stellar_spectra.values()),
                axis=0)
            stellar_spectrum = interpolator(T_star)
        else:
            stellar_spectrum = 1.0/self.lambda_grid**5/(np.exp(h*c/self.lambda_grid/k_B/T_star)-1)
        
        binned_wavelengths = []
        binned_depths = []
        for (start, end) in self.wavelength_bins:
            cond = np.logical_and(self.lambda_grid >= start, self.lambda_grid < end)
            binned_wavelengths.append(np.mean(self.lambda_grid[cond]))
            binned_depth = np.average(depths[cond], weights=stellar_spectrum[cond])
            binned_depths.append(binned_depth)
            
        return np.array(binned_wavelengths), np.array(binned_depths)

    def _validate_params(self, temperature, logZ, CO_ratio, cloudtop_pressure):
        
        if temperature is not None:
            minimum = max(np.min(self.T_grid), self.abundance_getter.min_temperature)
            maximum = np.max(self.T_grid)

            if temperature < minimum or temperature > maximum:
                raise ValueError("Temperature {} K is out of bounds ({} to {} K)".format(temperature, minimum, maximum))
        
        if logZ is not None:
            minimum = np.min(self.abundance_getter.logZs)
            maximum = np.max(self.abundance_getter.logZs)
            if logZ < minimum or logZ > maximum:
                raise ValueError("logZ {} is out of bounds ({} to {})".format(logZ, minimum, maximum))

        if CO_ratio is not None:
            minimum = np.min(self.abundance_getter.CO_ratios)
            maximum = np.max(self.abundance_getter.CO_ratios)
            if CO_ratio < minimum or CO_ratio > maximum:
                raise ValueError("C/O ratio {} is out of bounds ({} to {})".format(CO_ratio, minimum, maximum))

        if not np.isinf(cloudtop_pressure):
            minimum = self.min_P_profile
            maximum = self.max_P_profile
            if cloudtop_pressure <= minimum or cloudtop_pressure > maximum:
                raise ValueError("Cloudtop pressure is {} Pa, but must be between {} and {} Pa unless it is np.inf".format(cloudtop_pressure, minimum, maximum))                                                                       
                                 
    def compute_depths(self, star_radius, planet_mass, planet_radius,
                       temperature, logZ=0, CO_ratio=0.53,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope = 4, scattering_ref_wavelength = 1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None):
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
            radius of the planet at self.max_P_profile (by default,
            100,000 Pa).  Must be in metres.
        temperature : float
            Temperature of the isothermal atmosphere, in Kelvin
        logZ : float
            Base-10 logarithm of the metallicity, in solar units
        CO_ratio : float, optional
            C/O atomic ratio in the atmosphere.  The solar value is 0.53.
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
            abundance of Na at a pressure of self.P_grid[3] and temperature of
            self.T_grid[4].
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
       '''
        self._validate_params(temperature, logZ, CO_ratio, cloudtop_pressure)
        
        if custom_P_profile is not None:
            if custom_T_profile is None or len(custom_P_profile) != len(custom_T_profile):
                raise ValueError("Must specify both custom_T_profile and "\
                                 "custom_P_profile, and the two must have the"\
                                 " same length")
            if temperature is not None:
                raise ValueError("Cannot specify both temperature and custom T profile")
            P_profile = custom_P_profile
            T_profile = custom_T_profile
        else:    
            P_profile = np.logspace(
                np.log10(self.min_P_profile),
                np.log10(self.max_P_profile),
                self.num_profile_heights)
            T_profile = np.ones(len(P_profile)) * temperature

        abundances = self._get_abundances_array(logZ, CO_ratio, custom_abundances)
        above_clouds = P_profile < cloudtop_pressure

        radii, dr = self._get_above_cloud_r_and_dr(
            P_profile, T_profile, abundances, planet_mass, planet_radius,
            star_radius, above_clouds, T_star)
        
        P_profile = P_profile[above_clouds]
        T_profile = T_profile[above_clouds]

        T_cond = _interpolator_3D.get_condition_array(T_profile, self.T_grid)
        P_cond = _interpolator_3D.get_condition_array(P_profile, self.P_grid, cloudtop_pressure)

        absorption_coeff = self._get_gas_absorption(abundances, P_cond, T_cond)
        if add_scattering:
            absorption_coeff += self._get_scattering_absorption(
                abundances, P_cond, T_cond,
                scattering_factor, scattering_slope, scattering_ref_wavelength)

        if add_collisional_absorption:
            absorption_coeff += self._get_collisional_absorption(abundances, P_cond, T_cond)

        absorption_coeff_atm = _interpolator_3D.fast_interpolate(absorption_coeff, self.T_grid[T_cond], self.P_grid[P_cond], T_profile, P_profile)

        tau_los = get_line_of_sight_tau(absorption_coeff_atm, radii)

        absorption_fraction = 1 - np.exp(-tau_los)

        transit_depths = (planet_radius/star_radius)**2 + 2/star_radius**2 * absorption_fraction.dot(radii[1:] * dr)

        return self._get_binned_depths(transit_depths, T_star)
