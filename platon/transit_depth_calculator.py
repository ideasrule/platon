from pkg_resources import resource_filename

from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate

from ._compatible_loader import load_numpy_array
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from . import _interpolator_3D
from ._tau_calculator import get_line_of_sight_tau
from .constants import K_B, AMU, GM_SUN, TEFF_SUN

class TransitDepthCalculator:
    def __init__(self, star_radius, g, include_condensates=True, min_P_profile=0.1, max_P_profile=1e5, num_profile_heights=400):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        star_radius : float
            Radius of the star
        g : float
            Acceleration due to gravity of the planet at a pressure of
            max_P_profile
        include_condensates : bool
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

        self.star_radius = star_radius
        self.g = g

        self.absorption_data, self.mass_data, self.polarizability_data = read_species_data(
            resource_filename(__name__, "data/Absorption"),
            resource_filename(__name__, "data/species_info"))

        self.collisional_absorption_data = load_numpy_array(
            resource_filename(__name__, "data/collisional_absorption.pkl"))
        self.lambda_grid = load_numpy_array(
            resource_filename(__name__, "data/wavelengths.npy"))
        self.P_grid = load_numpy_array(
            resource_filename(__name__, "data/pressures.npy"))
        self.T_grid = load_numpy_array(
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

        self.abundance_getter = AbundanceGetter(include_condensates)

        self.min_P_profile = min_P_profile
        self.max_P_profile = max_P_profile
        self.num_profile_heights = num_profile_heights


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

        n = self.P_meshgrid[:,P_cond,:][:,:,T_cond]/(K_B*self.T_meshgrid[:,P_cond,:][:,:,T_cond])
        reshaped_lambda = self.lambda_grid.reshape((self.N_lambda, 1, 1))

        return multiple * (128.0/3 * np.pi**5) * n * sum_polarizability_sqr * ref_wavelength**(slope-4) / reshaped_lambda**slope


    def _get_collisional_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = np.zeros((self.N_lambda, np.sum(P_cond), np.sum(T_cond)))
        n = self.P_meshgrid[:,P_cond,:][:,:,T_cond]/(K_B * self.T_meshgrid[:,P_cond,:][:,:,T_cond])

        for s1, s2 in self.collisional_absorption_data:
            if s1 in abundances and s2 in abundances:
                n1 = (abundances[s1][P_cond, :][:,T_cond]*n)
                n2 = (abundances[s2][P_cond, :][:,T_cond]*n)
                abs_data = self.collisional_absorption_data[(s1,s2)].reshape((self.N_lambda, 1, self.N_T))[:,:,T_cond]
                absorption_coeff += abs_data * n1 * n2

        return absorption_coeff


    def _get_above_cloud_r_and_dr(self, P, T, abundances, planet_radius, P_cond):
        mu = np.zeros(len(P))
        for species_name in abundances:
            interpolator = RectBivariateSpline(self.P_grid, self.T_grid, abundances[species_name], kx=1, ky=1)
            atm_abundances = interpolator.ev(P, T)
            mu += atm_abundances * self.mass_data[species_name]

        atm_weight = UnivariateSpline(P,mu)
        T_profile = UnivariateSpline(P,T)
        GM = self.g*planet_radius**2

        R_hill = 0.5*self.star_radius*(TEFF_SUN/T[0])**2 * (GM/(3*GM_SUN))**(1/3)   #Hill radius for a sun like star

        if np.log(P[-1]/P[0]) > GM*mu[0]*AMU/(K_B*T[0])*(1/planet_radius - 1/R_hill):   #total number of scale heights required gives a radius that's larger than the hill radius
            print('The atmosphere is likely to be unbound. The scale height of the atmosphere is too large. Reverting to the constant g assumption')

            dP = P[1:] - P[0:-1]
            dr = dP/P[1:] * K_B * T[1:]/(mu[1:] * AMU * self.g)
            dr = np.append(K_B*T[0]/(mu[0] * AMU * self.g), dr)

            #dz goes from top to bottom of atmosphere
            radius_with_atm = np.sum(dr) + planet_radius
            radii = radius_with_atm - np.cumsum(dr)
            radii = np.append(radius_with_atm, radii[P_cond])

            return radii, dr[P_cond]

        def hydrostatic(y, P):
            r = y
            T_local = T_profile(P)
            local_mu = atm_weight(P)
            rho = local_mu*P*AMU / (K_B * T_local)
            dydP = r**2/(GM * rho)
            return dydP

        y0 = planet_radius

        radii_ode = np.transpose(integrate.odeint(hydrostatic,y0,P))[0]
        dr = np.diff(radii_ode)
        dr = np.flipud(np.append(dr,K_B*T[0]/(mu[0] * AMU * self.g)))
        radius_with_atm = planet_radius + np.sum(dr)
        radii = radius_with_atm - np.cumsum(dr)
        radii = np.append(radius_with_atm, radii[P_cond])

        return radii, dr[P_cond]

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

    def is_in_bounds(self, logZ, CO_ratio, T, cloudtop_P):
        '''Tests whether a certain combination of parameters is within the
        bounds of the data files. The arguments are the same as those in
        `compute_depths.`'''

        if T <= np.min(self.T_grid) or T >= np.max(self.T_grid): return False
        if cloudtop_P <= self.min_P_profile or cloudtop_P >= self.max_P_profile: return False
        return self.abundance_getter.is_in_bounds(logZ, CO_ratio, T)

    def compute_depths(self, planet_radius, temperature, logZ=0, CO_ratio=0.53,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope = 4, scattering_ref_wavelength = 1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None):
        '''
        Computes transit depths at a range of wavelengths, assuming an
        isothermal atmosphere.  To choose bins, call change_wavelength_bins().

        Parameters
        ----------
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
        Returns
        -------
        wavelengths : array of float
            Central wavelengths, in metres
        transit_depths : array of float
            Transit depths at `wavelengths`
       '''

        P_profile = np.logspace(np.log10(self.min_P_profile), np.log10(self.max_P_profile), self.num_profile_heights)
        T_profile = np.ones(len(P_profile)) * temperature

        abundances = self._get_abundances_array(logZ, CO_ratio, custom_abundances)
        above_clouds = P_profile < cloudtop_pressure
        radii, dr = self._get_above_cloud_r_and_dr(P_profile, T_profile, abundances, planet_radius, above_clouds)
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

        transit_depths = (planet_radius/self.star_radius)**2 + 2/self.star_radius**2 * absorption_fraction.dot(radii[1:] * dr)

        binned_wavelengths = []
        binned_depths = []
        if self.wavelength_bins is not None:
            for (start, end) in self.wavelength_bins:
                cond = np.logical_and(self.lambda_grid >= start, self.lambda_grid < end)
                binned_wavelengths.append(np.mean(self.lambda_grid[cond]))
                binned_depths.append(np.mean(transit_depths[cond]))
            return np.array(binned_wavelengths), np.array(binned_depths)

        return self.lambda_grid, transit_depths


