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
from ._loader import load_dict_from_pickle, load_numpy
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from . import _interpolator_3D
from ._tau_calculator import get_line_of_sight_tau
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from ._get_data import get_data_if_needed
from ._mie_cache import MieCache
from .errors import AtmosphereError


class AtmosphereSolver:
    def __init__(self, include_condensation=True, num_profile_heights=250,
                 ref_pressure=1e5, method='xsec'):
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
        '''
        self.arguments = locals()
        del self.arguments["self"]

        get_data_if_needed()
        
        self.absorption_data, self.mass_data, self.polarizability_data = read_species_data(
            resource_filename(__name__, "data/Absorption"),
            resource_filename(__name__, "data/species_info"),
            method)
        #self.absorption_data["H2O"] *= 0

        self.low_res_lambdas = load_numpy("data/low_res_lambdas.npy")
        self.collisional_absorption_data = load_dict_from_pickle(
            "data/collisional_absorption.pkl")

        if method == "xsec":
            self.lambda_grid = load_numpy("data/wavelengths.npy")
            self.d_ln_lambda = np.median(np.diff(np.log(self.lambda_grid)))
            self.stellar_spectra = load_dict_from_pickle("data/stellar_spectra.pkl")
        else:
            self.lambda_grid = load_numpy("data/k_wavelengths.npy")
            diffs = np.unique(self.lambda_grid)
            self.d_ln_lambda = np.median(np.diff(np.log(np.unique(self.lambda_grid))))
            self.stellar_spectra = load_dict_from_pickle("data/k_stellar_spectra.pkl")
        
        self.P_grid = load_numpy("data/pressures.npy")
        self.T_grid = load_numpy("data/temperatures.npy")

        self.N_lambda = len(self.lambda_grid)
        self.N_T = len(self.T_grid)
        self.N_P = len(self.P_grid)

        self.wavelength_rebinned = False
        self.wavelength_bins = None

        self.abundance_getter = AbundanceGetter(include_condensation)
        self.min_temperature = max(np.min(self.T_grid), self.abundance_getter.min_temperature)
        self.max_temperature = np.max(self.T_grid)

        self.num_profile_heights = num_profile_heights
        self.ref_pressure = ref_pressure
        self.method = method
        self._mie_cache = MieCache()

        self.all_cross_secs = load_dict_from_pickle("data/all_cross_secs.pkl")
        self.all_radii = load_numpy("data/mie_radii.npy")


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
        if self.wavelength_rebinned:
            self.__init__(**self.arguments)
            self.wavelength_rebinned = False        
            
        if bins is None:
            return

        for start, end in bins:
            if start < np.min(self.lambda_grid) \
               or start > np.max(self.lambda_grid) \
               or end < np.min(self.lambda_grid) \
               or end > np.max(self.lambda_grid):
                raise ValueError("Invalid wavelength bin: {}-{} meters".format(start, end))
        
        self.wavelength_rebinned = True
        self.wavelength_bins = bins

        cond = np.any([
            np.logical_and(self.lambda_grid > start, self.lambda_grid < end) \
            for (start, end) in bins], axis=0)

        for key in self.absorption_data:
            self.absorption_data[key] = self.absorption_data[key][:, :, cond]

        self.lambda_grid = self.lambda_grid[cond]
        self.N_lambda = len(self.lambda_grid)

        for Teff in self.stellar_spectra:
            self.stellar_spectra[Teff] = self.stellar_spectra[Teff][cond]


    def _get_gas_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = np.zeros(
            (np.sum(T_cond), np.sum(P_cond), self.N_lambda))
        
        for species_name, species_abundance in abundances.items():
            assert(species_abundance.shape == (self.N_T, self.N_P))
            
            if species_name in self.absorption_data:
                absorption_coeff += self.absorption_data[species_name][T_cond][:,P_cond] * species_abundance[T_cond][:,P_cond,np.newaxis]

        return absorption_coeff

    def _get_scattering_absorption(self, abundances, P_cond, T_cond,
                                   multiple=1, slope=4, ref_wavelength=1e-6):
        sum_polarizability_sqr = np.zeros((np.sum(T_cond), np.sum(P_cond)))

        for species_name in abundances:
            if species_name in self.polarizability_data:
                sum_polarizability_sqr += abundances[species_name][T_cond,:][:,P_cond] * self.polarizability_data[species_name]**2

        n = self.P_grid[P_cond] / (k_B * self.T_grid[T_cond][:, np.newaxis])
        result = (multiple * (128.0 / 3 * np.pi**5) * ref_wavelength**(slope - 4) * n * sum_polarizability_sqr)[:, :, np.newaxis] / self.lambda_grid**slope
        return result

    def _get_collisional_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = np.zeros(
            (np.sum(T_cond), np.sum(P_cond), self.N_lambda))
        n = self.P_grid[np.newaxis, P_cond] / (k_B * self.T_grid[T_cond, np.newaxis])
        no_interp = len(self.lambda_grid) == len(self.low_res_lambdas) and np.allclose(self.lambda_grid, self.low_res_lambdas)
        
        for s1, s2 in self.collisional_absorption_data:
            if s1 in abundances and s2 in abundances:
                n1 = (abundances[s1][T_cond, :][:, P_cond] * n)
                n2 = (abundances[s2][T_cond, :][:, P_cond] * n)
                abs_data = self.collisional_absorption_data[(s1, s2)].reshape(
                    (self.N_T, 1, len(self.low_res_lambdas)))[T_cond]

                if no_interp:
                    absorption_coeff += abs_data * (n1 * n2)[:, :, np.newaxis]
                else:
                    absorption_coeff += scipy.interpolate.interp1d(
                        self.low_res_lambdas,
                        abs_data * (n1 * n2)[:, :, np.newaxis])(self.lambda_grid)

        return absorption_coeff

    def _get_mie_scattering_absorption(self, P_cond, T_cond, ri, part_size,
                                       frac_scale_height, max_number_density,
                                       sigma = 0.5, max_zscore = 5, num_integral_points = 100):
        if isinstance(ri, str):
            eff_cross_section = np.interp(
                self.lambda_grid,
                self.low_res_lambdas,
                scipy.interpolate.interp1d(self.all_radii, self.all_cross_secs[ri])(part_size))
        else:
            eff_cross_section = np.zeros(self.N_lambda)
            z_scores = -np.logspace(np.log10(0.1), np.log10(max_zscore), num_integral_points/2)
            z_scores = np.append(z_scores[::-1], -z_scores)

            probs = np.exp(-z_scores**2/2) / np.sqrt(2 * np.pi)
            radii = part_size * np.exp(z_scores * sigma)
            geometric_cross_section = np.pi * radii**2

            dense_xs = 2*np.pi*radii[np.newaxis,:] / self.lambda_grid[:,np.newaxis]
            dense_xs = dense_xs.flatten()

            x_hist = np.histogram(dense_xs, bins='auto')[1]
            Qext_hist = self._mie_cache.get_and_update(ri, x_hist) 

            spl = scipy.interpolate.splrep(x_hist, Qext_hist)
            Qext_intpl = scipy.interpolate.splev(dense_xs, spl)
            Qext_intpl = np.reshape(Qext_intpl, (self.N_lambda, len(radii)))

            eff_cross_section = np.trapz(probs*geometric_cross_section*Qext_intpl, z_scores)

        n = max_number_density * np.power(self.P_grid[P_cond] / max(self.P_grid[P_cond]), 1.0/frac_scale_height)
        absorption_coeff = n[np.newaxis, :, np.newaxis] * eff_cross_section[np.newaxis, np.newaxis, :]
        
        return absorption_coeff

    def _get_above_cloud_profiles(self, P_profile, T_profile, abundances,
                                  planet_mass, planet_radius, star_radius,
                                  above_cloud_cond, T_star=None):
        
        assert(len(P_profile) == len(T_profile))
        # First, get atmospheric weight profile
        mu_profile = np.zeros(len(P_profile))
        atm_abundances = {}
        
        for species_name in abundances:
            interpolator = RectBivariateSpline(
                self.T_grid, np.log10(self.P_grid),
                np.log10(abundances[species_name]), kx=1, ky=1)
            abund = 10**interpolator.ev(T_profile, np.log10(P_profile))
            atm_abundances[species_name] = abund
            mu_profile += abund * self.mass_data[species_name]

        radii, dr = _hydrostatic_solver._solve(
            P_profile, T_profile, self.ref_pressure, mu_profile, planet_mass,
            planet_radius, star_radius, above_cloud_cond, T_star)
        
        for key in atm_abundances:
            atm_abundances[key] = atm_abundances[key][above_cloud_cond]
            
        return radii, dr, atm_abundances, mu_profile

    def _get_abundances_array(self, logZ, CO_ratio, custom_abundances):
        if custom_abundances is None:
            return self.abundance_getter.get(logZ, CO_ratio)

        if logZ is not None or CO_ratio is not None:
            raise ValueError(
                "Must set logZ=None and CO_ratio=None to use custom_abundances")

        if isinstance(custom_abundances, str):
            # Interpret as filename
            return AbundanceGetter.from_file(custom_abundances)

        if isinstance(custom_abundances, dict):
            for key, value in custom_abundances.items():
                if not isinstance(value, np.ndarray):
                    raise ValueError(
                        "custom_abundances must map species names to arrays")
                if value.shape != (self.N_T, self.N_P):
                    raise ValueError(
                        "custom_abundances has array of invalid size")
            return custom_abundances

        raise ValueError("Unrecognized format for custom_abundances")
   

    def _validate_params(self, T_profile, logZ, CO_ratio, cloudtop_pressure):
        if np.min(T_profile) < self.min_temperature or\
           np.max(T_profile) > self.max_temperature:
            raise AtmosphereError("Invalid temperatures in T/P profile")
            
        if logZ is not None:
            minimum = np.min(self.abundance_getter.logZs)
            maximum = np.max(self.abundance_getter.logZs)
            if logZ < minimum or logZ > maximum:
                raise ValueError(
                    "logZ {} is out of bounds ({} to {})".format(
                        logZ, minimum, maximum))

        if CO_ratio is not None:
            minimum = np.min(self.abundance_getter.CO_ratios)
            maximum = np.max(self.abundance_getter.CO_ratios)
            if CO_ratio < minimum or CO_ratio > maximum:
                raise ValueError(
                    "C/O ratio {} is out of bounds ({} to {})".format(CO_ratio, minimum, maximum))

        if not np.isinf(cloudtop_pressure):
            minimum = np.min(self.P_grid)
            maximum = np.max(self.P_grid)
            if cloudtop_pressure <= minimum or cloudtop_pressure > maximum:
                raise ValueError(
                    "Cloudtop pressure is {} Pa, but must be between {} and {} Pa unless it is np.inf".format(
                        cloudtop_pressure, minimum, maximum))

    def get_stellar_spectrum(self, lambdas, T_star, T_spot, spot_cov_frac, blackbody=False):
        if spot_cov_frac is None:
            spot_cov_frac = 0

        if T_spot is None:
            T_spot = T_star
            
        temperatures = list(self.stellar_spectra.keys()) #NOT sorted!
        if T_star is None:
            unspotted_spectrum = np.ones(len(lambdas))
            spot_spectrum = np.ones(len(lambdas))
        elif T_star >= np.min(temperatures) and T_star <= np.max(temperatures) and not blackbody:
            interpolator = scipy.interpolate.interp1d(
                temperatures, list(self.stellar_spectra.values()),
                assume_sorted=False,
                axis=0)
            unspotted_spectrum = interpolator(T_star)
            spot_spectrum = interpolator(T_spot)
        else:
            d_lambda = self.d_ln_lambda * lambdas
            unspotted_spectrum = 2 * c * np.pi / lambdas**4 / \
                (np.exp(h * c / lambdas / k_B / T_star) - 1) * d_lambda
            spot_spectrum = 2 * c * np.pi / lambdas**4 / \
                (np.exp(h * c / lambdas / k_B / T_spot) - 1) * d_lambda

        stellar_spectrum = spot_cov_frac * spot_spectrum + \
                           (1 - spot_cov_frac) * unspotted_spectrum
        correction_factors = unspotted_spectrum/stellar_spectrum
        return stellar_spectrum, correction_factors
            
    def compute_params(self, star_radius, planet_mass, planet_radius,
                       P_profile, T_profile,
                       logZ=0, CO_ratio=0.53,
                       add_gas_absorption=True,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5,
                       P_quench=1e-99,
                       min_abundance=1e-99, min_cross_sec=1e-99):       
        self._validate_params(T_profile, logZ, CO_ratio, cloudtop_pressure)
       
        abundances = self._get_abundances_array(
            logZ, CO_ratio, custom_abundances)
        
        for name in abundances:
            low_abundances = abundances[name] < min_abundance
            abundances[name][low_abundances] = min_abundance
            for t in range(abundances[name].shape[0]):
                quench_abund = np.exp(np.interp(np.log(P_quench), np.log(self.P_grid), np.log(abundances[name][t])))
                abundances[name][t, self.P_grid < P_quench] = quench_abund
            
        above_clouds = P_profile < cloudtop_pressure

        radii, dr, atm_abundances, mu_profile = self._get_above_cloud_profiles(
            P_profile, T_profile, abundances, planet_mass, planet_radius,
            star_radius, above_clouds, T_star)

        P_profile = P_profile[above_clouds]
        T_profile = T_profile[above_clouds]

        T_cond = _interpolator_3D.get_condition_array(T_profile, self.T_grid)
        P_cond = _interpolator_3D.get_condition_array(
            P_profile, self.P_grid, cloudtop_pressure)
        absorption_coeff = np.zeros((np.sum(T_cond), np.sum(P_cond), len(self.lambda_grid)))
        if add_gas_absorption:
            absorption_coeff += self._get_gas_absorption(abundances, P_cond, T_cond)
        if add_scattering:
            if ri is not None:
                if scattering_factor != 1 or scattering_slope != 4:
                    raise ValueError("Cannot use both parametric and Mie scattering at the same time")
                
                absorption_coeff += self._get_mie_scattering_absorption(
                    P_cond, T_cond, ri, part_size,
                    frac_scale_height, number_density, sigma=part_size_std)
                absorption_coeff += self._get_scattering_absorption(abundances,
                P_cond, T_cond)
                
            else:
                absorption_coeff += self._get_scattering_absorption(abundances,
                P_cond, T_cond, scattering_factor, scattering_slope,
                scattering_ref_wavelength)

        if add_collisional_absorption:
            absorption_coeff += self._get_collisional_absorption(
                abundances, P_cond, T_cond)

        # Cross sections vary less than absorption coefficients by pressure
        # and temperature, so interpolation should be done with cross sections
        cross_secs = absorption_coeff / (self.P_grid[P_cond][np.newaxis, :, np.newaxis] / k_B / self.T_grid[T_cond][:, np.newaxis, np.newaxis])
        cross_secs[cross_secs < min_cross_sec] = min_cross_sec
        
        if len(self.T_grid[T_cond]) == 1:
            cross_secs_atm = np.exp(scipy.interpolate.interpn((np.log(self.P_grid[P_cond]),), np.log(cross_secs[0]), np.log(P_profile)))
        else:
            # log(sigma) goes linearly with 1/T more than with T
            cross_secs_atm = np.exp(scipy.interpolate.interpn(
                (1.0/self.T_grid[T_cond][::-1], np.log(self.P_grid[P_cond])),
                np.log(cross_secs[::-1]),
                np.array([1.0/T_profile, np.log(P_profile)]).T))

        absorption_coeff_atm = cross_secs_atm * (P_profile / k_B / T_profile)[:, np.newaxis]
        output_dict = {"absorption_coeff_atm": absorption_coeff_atm,
                       "radii": radii,
                       "dr": dr,
                       "P_profile": P_profile,
                       "T_profile": T_profile,
                       "mu_profile": mu_profile,
                       "atm_abundances": atm_abundances,
                       "unbinned_wavelengths": self.lambda_grid}
        
        return output_dict
