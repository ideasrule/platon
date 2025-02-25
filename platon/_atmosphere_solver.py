import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import warnings

from pkg_resources import resource_filename
from ._hist import get_num_bins
from ._interpolator_3D import interp1d
from . import _cupy_numpy as xp
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
from ._interpolator_3D import regular_grid_interp, interp1d

class AtmosphereSolver:
    def __init__(self, include_condensation=True, ref_pressure=1e5, method='xsec', include_opacities=[], downsample=1):
        self.arguments = locals()
        del self.arguments["self"]

        get_data_if_needed()
        
        self.absorption_data, self.mass_data, self.polarizability_data = read_species_data(
            resource_filename(__name__, "data/Absorption"),
            resource_filename(__name__, "data/species_info"),
            method, include_opacities, downsample)

        self.low_res_lambdas = load_numpy("data/low_res_lambdas.npy")
        self.stellar_spectra_dict = load_dict_from_pickle("data/stellar_spectra.pkl")                    
        
        if method == "xsec":
            self.lambda_grid = xp.copy(load_numpy("data/wavelengths.npy")[::downsample])
            self.d_ln_lambda = xp.median(xp.diff(xp.log(self.lambda_grid)))
        else:
            warnings.warn("Correlated-k is not recommended, and will probably be removed in a future version! Please use xsec")
            self.lambda_grid = load_numpy("data/k_wavelengths.npy")
            self.d_ln_lambda = xp.median(xp.diff(xp.log(xp.unique(self.lambda_grid))))

        self.orig_lambda_grid = xp.copy(self.lambda_grid) #In case we truncate with change_wavelength_bins
        self.stellar_spectra_temps = xp.array(self.stellar_spectra_dict["temperatures"])
        self.stellar_spectra = list(self.stellar_spectra_dict["spectra"])
        for i in range(len(self.stellar_spectra)):
            self.stellar_spectra[i] = xp.interp(self.lambda_grid, self.low_res_lambdas, self.stellar_spectra[i])
        self.stellar_spectra = xp.array(self.stellar_spectra)
            
        self.collisional_absorption_data = load_dict_from_pickle(
            "data/collisional_absorption.pkl")
        
        for key in self.collisional_absorption_data:
            val = interp1d(self.lambda_grid, self.low_res_lambdas, self.collisional_absorption_data[key].T).T
            self.collisional_absorption_data[key] = xp.copy(val, order="C")
            
        self.P_grid = load_numpy("data/pressures.npy")
        self.T_grid = load_numpy("data/temperatures.npy")

        self.N_lambda = len(self.lambda_grid)
        self.N_T = len(self.T_grid)
        self.N_P = len(self.P_grid)

        self.wavelength_rebinned = False
        self.wavelength_bins = None

        self.abundance_getter = AbundanceGetter(include_condensation)
        self.min_temperature = max(self.T_grid.min(), self.abundance_getter.min_temperature)
        self.max_temperature = xp.amax(self.T_grid)

        self.ref_pressure = ref_pressure
        self.method = method
        self._mie_cache = MieCache()

        self.all_cross_secs = load_dict_from_pickle("data/all_cross_secs.pkl")
        self.all_radii = load_numpy("data/mie_radii.npy")
        diffs = np.diff(np.log(self.all_radii))
        self.d_ln_radii = np.median(diffs)
        assert(np.allclose(diffs, self.d_ln_radii))

    def get_lambda_grid(self):
        return xp.cpu(self.lambda_grid)
        
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
        bins = xp.array(bins)
        if self.wavelength_rebinned:
            self.__init__(**self.arguments)
            self.wavelength_rebinned = False        
            
        if bins is None:
            return

        for start, end in bins:
            if start < self.lambda_grid.min() \
               or start > self.lambda_grid.max() \
               or end < self.lambda_grid.min() \
               or end > self.lambda_grid.max():
                raise ValueError("Invalid wavelength bin: {}-{} meters".format(start, end))
            num_points = xp.sum(xp.logical_and(self.lambda_grid > start,
                                               self.lambda_grid < end))
            if num_points == 0:
                raise ValueError("Wavelength bin too narrow: {}-{} meters".format(start, end))
            if num_points <= 5:
                print("WARNING: only {} points in {}-{} m bin. Results will be inaccurate".format(num_points, start, end))
        
        self.wavelength_rebinned = True
        self.wavelength_bins = bins

        cond = xp.any(xp.array([
            xp.logical_and(self.lambda_grid > start, self.lambda_grid < end) \
            for (start, end) in bins]), axis=0)

        for key in self.absorption_data:
            self.absorption_data[key] = self.absorption_data[key][:, :, cond]

        self.lambda_grid = self.lambda_grid[cond]
        self.N_lambda = len(self.lambda_grid)

        self.stellar_spectra = self.stellar_spectra[:,cond]
            
        for key in self.collisional_absorption_data:
            self.collisional_absorption_data[key] = self.collisional_absorption_data[key][:, cond]
            
    def _get_k(self, T, wavelengths):
        wavelengths = 1e6 * xp.copy(wavelengths)
        alpha = 14391
        lambda_0 = 1.6419

        #Calculate bound-free absorption coefficient
        k_bf = xp.zeros(len(wavelengths))
        cond = wavelengths < lambda_0
        C = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
        f_lambda = xp.sum(xp.array([C[i-1] * (1/wavelengths[cond] - 1/lambda_0)**((i-1)/2) for i in range(1,7)]), axis=0)
        sigma = 1e-18 * wavelengths[cond]**3 * (1 / wavelengths[cond] - 1 / lambda_0)**1.5 * f_lambda
        k_bf[cond] = 0.75 * T**-2.5 * xp.exp(alpha/lambda_0 / T) * (1 - xp.exp(-alpha / wavelengths[cond] / T)) * sigma

        #Now calculate free-free absorption coefficient
        k_ff = xp.zeros(len(wavelengths))
        mid = xp.logical_and(wavelengths > 0.1823, wavelengths < 0.3645)
        red = wavelengths > 0.3645
                    
        ff_matrix_red = xp.array([
            [0, 0, 0, 0, 0, 0],
            [2483.346, 285.827, -2054.291, 2827.776, -1341.537, 208.952],
            [-3449.889, -1158.382, 8746.523, -11485.632, 5303.609, -812.939],
            [2200.04, 2427.719, -13651.105, 16755.524, -7510.494, 1132.738],
            [-696.271, -1841.4, 8624.97, -10051.53, 4400.067, -655.02],
            [88.283, 444.517, -1863.864, 2095.288, -901.788, 132.985]])
        ff_matrix_mid = xp.array([
            [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
            [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.36],
            [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
            [115.5291, 169.6374, -245.649, 114.243, -21.9972, 1.5097],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])

        for n in range(1, 7):
            A_mid = xp.array([wavelengths[mid]**i for i in (2, 0, -1, -2, -3, -4)]).T
            #print(A_mid.shape)
            A_red = xp.array([wavelengths[red]**i for i in (2, 0, -1, -2, -3, -4)]).T
            
            k_ff[mid] += 1e-29 * (5040/T)**((n+1)/2) * A_mid.dot(ff_matrix_mid[n-1])
            k_ff[red] += 1e-29 * (5040/T)**((n+1)/2) * A_red.dot(ff_matrix_red[n-1])

        k = k_bf + k_ff
        
        #1e-3 to convert from cm^4/dyne to m^4/N
        return k * 1e-3
    
    def _get_H_minus_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = xp.zeros(
            (int(xp.sum(T_cond)), int(xp.sum(P_cond)), self.N_lambda))
        
        valid_Ts = self.T_grid[T_cond]
        trunc_el_abundances = abundances["el"][T_cond][:, P_cond]
        trunc_H_abundances = abundances["H"][T_cond][:, P_cond]
        
        for t in range(len(valid_Ts)):
            k = self._get_k(valid_Ts[t], self.lambda_grid)          
            absorption_coeff[t] = k * (trunc_el_abundances[t] * trunc_H_abundances[t] * self.P_grid[P_cond]**2)[:, xp.newaxis] / (k_B * valid_Ts[t])
                  
        return absorption_coeff

    def _get_gas_absorption(self, abundances, P_cond, T_cond, zero_opacities=[]):
        absorption_coeff = xp.zeros(
            (int(xp.sum(T_cond)), int(xp.sum(P_cond)), self.N_lambda))

        for species_name, species_abundance in abundances.items():
            assert(species_abundance.shape == (self.N_T, self.N_P))
            
            if (species_name in self.absorption_data) and (species_name not in zero_opacities):
                absorption_coeff += self.absorption_data[species_name][T_cond][:,P_cond] * species_abundance[T_cond][:,P_cond,xp.newaxis]

        return absorption_coeff

    def _get_scattering_absorption(self, abundances, P_cond, T_cond,
                                   multiple=1, slope=4, ref_wavelength=1e-6):
        sum_polarizability_sqr = xp.zeros((int(xp.sum(T_cond)), int(xp.sum(P_cond))))

        for species_name in abundances:
            if species_name in self.polarizability_data:
                sum_polarizability_sqr += abundances[species_name][T_cond,:][:,P_cond] * self.polarizability_data[species_name]**2

        n = self.P_grid[P_cond] / (k_B * self.T_grid[T_cond][:, xp.newaxis])
        result = (multiple * (128.0 / 3 * xp.pi**5) * ref_wavelength**(slope - 4) * n * sum_polarizability_sqr)[:, :, xp.newaxis] / self.lambda_grid**slope
        return result

    def _get_collisional_absorption(self, abundances, P_cond, T_cond):
        absorption_coeff = xp.zeros(
            (int(xp.sum(T_cond)), int(xp.sum(P_cond)), self.N_lambda))
        
        n = self.P_grid[xp.newaxis, P_cond] / (k_B * self.T_grid[T_cond, xp.newaxis])        
        for s1, s2 in self.collisional_absorption_data:
            if s1 in abundances and s2 in abundances:
                n1 = (abundances[s1][T_cond, :][:, P_cond] * n)
                n2 = (abundances[s2][T_cond, :][:, P_cond] * n)
                abs_data = self.collisional_absorption_data[(s1, s2)].reshape(
                    (self.N_T, 1, -1))[T_cond]
                absorption_coeff += abs_data * (n1 * n2)[:, :, xp.newaxis]

        return absorption_coeff

    def _get_mie_scattering_absorption(self, P_cond, T_cond, ri, part_size,
                                       frac_scale_height, max_number_density,
                                       sigma = 0.5, max_zscore = 5, num_integral_points = 100):
        if isinstance(ri, str):
            kernel = sigma / self.d_ln_radii
            cross_secs = xp.ndimage.gaussian_filter(self.all_cross_secs[ri], kernel)
            if part_size < self.all_radii[3*int(kernel)] or part_size > self.all_radii[-3*int(kernel)]:
                raise ValueError("part_size out of bounds: {} m".format(part_size))
            
            eff_cross_section = xp.interp(
                self.lambda_grid,
                self.low_res_lambdas,
                interp1d(part_size, self.all_radii, cross_secs.T))
        else:
            eff_cross_section = xp.zeros(self.N_lambda)
            z_scores = -xp.logspace(xp.log10(0.1), xp.log10(max_zscore), int(num_integral_points/2))
            z_scores = xp.append(z_scores[::-1], -z_scores)

            probs = xp.exp(-z_scores**2/2) / xp.sqrt(2 * xp.pi)
            radii = part_size * xp.exp(z_scores * sigma)
            geometric_cross_section = xp.pi * radii**2

            dense_xs = 2*xp.pi*radii[xp.newaxis,:] / self.lambda_grid[:,xp.newaxis]
            log_dense_xs = xp.log(dense_xs.flatten())

            n_bins = get_num_bins(log_dense_xs)
            log_x_hist = xp.cpu(xp.histogram(log_dense_xs, bins=n_bins)[1])
            
            Qext_hist = self._mie_cache.get_and_update(ri, np.exp(log_x_hist))
            spl = scipy.interpolate.make_interp_spline(log_x_hist, Qext_hist)
            spl = xp.interpolate.BSpline(xp.array(spl.t), xp.array(spl.c), spl.k)           
            Qext_intpl = spl(log_dense_xs).reshape((self.N_lambda, len(radii)))
            eff_cross_section = xp.trapz(probs*geometric_cross_section*Qext_intpl, z_scores)

        n = max_number_density * xp.power(self.P_grid[P_cond] / max(self.P_grid[P_cond]), 1.0/frac_scale_height)        
        absorption_coeff = n[xp.newaxis, :, xp.newaxis] * eff_cross_section[xp.newaxis, xp.newaxis, :]
        return absorption_coeff

    def _get_above_cloud_profiles(self, P_profile, T_profile, abundances,
                                  planet_mass, planet_radius, star_radius,
                                  above_cloud_cond, T_star=None):        
        assert(len(P_profile) == len(T_profile))
        # First, get atmospheric weight profile
        mu_profile = xp.zeros(len(P_profile))
        atm_abundances = {}
        
        for species_name in abundances:
            abund = 10.**regular_grid_interp(self.T_grid, xp.log10(self.P_grid), xp.log10(abundances[species_name]), T_profile, xp.log10(P_profile))
            atm_abundances[species_name] = abund
            mu_profile += abund * self.mass_data[species_name]

        radii, dr = _hydrostatic_solver._solve(
            P_profile, T_profile, self.ref_pressure, mu_profile, planet_mass,
            planet_radius, star_radius, above_cloud_cond, T_star)
        
        for key in atm_abundances:
            atm_abundances[key] = atm_abundances[key][above_cloud_cond]
            
        return radii, dr, atm_abundances, mu_profile

    def _get_abundances_array(self, logZ, CO_ratio, CH4_mult, custom_abundances, gases, vmrs):
        if custom_abundances is None and logZ is not None and CO_ratio is not None:
            abunds = self.abundance_getter.get(logZ, CO_ratio)
            abunds["CH4"] *= CH4_mult
            return abunds

        if logZ is not None or CO_ratio is not None:
            raise ValueError(
                "Must set logZ=None and CO_ratio=None to use custom_abundances")

        if isinstance(custom_abundances, str):
            # Interpret as filename
            return AbundanceGetter.from_file(custom_abundances)

        if isinstance(custom_abundances, dict):
            for key, value in custom_abundances.items():
                if not isinstance(value, xp.ndarray):
                    raise ValueError(
                        "custom_abundances must map species names to arrays")
                if value.shape != (self.N_T, self.N_P):
                    raise ValueError(
                        "custom_abundances has array of invalid size")
            return custom_abundances

        
        if custom_abundances is None and vmrs is not None and gases is not None:
            abundances = {}
            for i, g in enumerate(gases):
                abundances[g] = vmrs[i] * xp.ones((len(self.T_grid), len(self.P_grid)))
            return abundances

        raise ValueError("Unrecognized format for custom_abundances")
   

    def _validate_params(self, T_profile, logZ, CO_ratio, cloudtop_pressure):
        T_profile = xp.atleast_1d(T_profile)
        if T_profile.min() < self.min_temperature or\
           T_profile.max() > self.max_temperature:
            raise AtmosphereError("Invalid temperatures in T/P profile")
            
        if logZ is not None:
            minimum = self.abundance_getter.logZs.min()
            maximum = self.abundance_getter.logZs.max()
            if logZ < minimum or logZ > maximum:
                raise ValueError(
                    "logZ {} is out of bounds ({} to {})".format(
                        logZ, minimum, maximum))

        if CO_ratio is not None:
            minimum = self.abundance_getter.CO_ratios.min()
            maximum = self.abundance_getter.CO_ratios.max()
            if CO_ratio < minimum or CO_ratio > maximum:
                raise ValueError(
                    "C/O ratio {} is out of bounds ({} to {})".format(CO_ratio, minimum, maximum))

        if not xp.isinf(cloudtop_pressure):
            minimum = self.P_grid.min()
            maximum = self.P_grid.max()
            if cloudtop_pressure <= minimum or cloudtop_pressure > maximum:
                raise ValueError(
                    "Cloudtop pressure is {} Pa, but must be between {} and {} Pa unless it is xp.inf".format(
                        cloudtop_pressure, minimum, maximum))

    def get_stellar_spectrum(self, lambdas, T_star, T_spot, spot_cov_frac, blackbody=False):
        if spot_cov_frac is None:
            spot_cov_frac = 0

        if T_spot is None:
            T_spot = T_star
            
        if T_star is None:
            unspotted_spectrum = xp.ones(len(lambdas))
            spot_spectrum = xp.ones(len(lambdas))
            
        elif T_star >= self.stellar_spectra_temps.min() and T_star <= self.stellar_spectra_temps.max() and not blackbody:            
            unspotted_spectrum = interp1d(T_star, self.stellar_spectra_temps, self.stellar_spectra)
            spot_spectrum = interp1d(T_spot, self.stellar_spectra_temps, self.stellar_spectra)
            if len(spot_spectrum) != len(lambdas):
                raise ValueError("Stellar spectra has a different length ({}) than opacities ({})!  If you are using high resolution opacities, pass stellar_blackbody=True to compute_depths".format(len(spot_spectrum), len(lambdas)))
        else:
            d_lambda = self.d_ln_lambda * lambdas
            unspotted_spectrum = 2 * c * xp.pi / lambdas**4 / \
                (xp.exp(h * c / lambdas / k_B / T_star) - 1) * h * c / lambdas
            spot_spectrum = 2 * c * xp.pi / lambdas**4 / \
                (xp.exp(h * c / lambdas / k_B / T_spot) - 1) * h * c / lambdas

        stellar_spectrum = spot_cov_frac * spot_spectrum + \
                           (1 - spot_cov_frac) * unspotted_spectrum
        correction_factors = unspotted_spectrum/stellar_spectrum
        return stellar_spectrum, correction_factors

    def compute_params(self, star_radius, planet_mass, planet_radius,
                       P_profile, T_profile,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       add_gas_absorption=True,
                       add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf, custom_abundances=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5,
                       P_quench=1e-99,
                       min_abundance=1e-99, min_cross_sec=1e-99, zero_opacities=[]):
        self._validate_params(T_profile, logZ, CO_ratio, cloudtop_pressure)
       
        abundances = self._get_abundances_array(
            logZ, CO_ratio, CH4_mult, custom_abundances, gases, vmrs)

        T_quench = xp.interp(xp.log(P_quench), xp.log(P_profile), T_profile)
        for name in abundances:
            abundances[name][xp.isnan(abundances[name])] = min_abundance
            abundances[name][abundances[name] < min_abundance] = min_abundance
            quench_abund = 10.**regular_grid_interp(self.T_grid, xp.log10(self.P_grid), xp.log10(abundances[name]), T_quench, xp.log10(P_quench))
            abundances[name][:, self.P_grid <= P_quench] = quench_abund

        above_clouds = P_profile < cloudtop_pressure

        radii, dr, atm_abundances, mu_profile = self._get_above_cloud_profiles(
            P_profile, T_profile, abundances, planet_mass, planet_radius,
            star_radius, above_clouds, T_star)
            
        P_profile = P_profile[above_clouds]
        T_profile = T_profile[above_clouds]

        T_cond = _interpolator_3D.get_condition_array(T_profile, self.T_grid)
        P_cond = _interpolator_3D.get_condition_array(
            P_profile, self.P_grid, cloudtop_pressure)

        absorption_coeff = xp.zeros((int(xp.sum(T_cond)), int(xp.sum(P_cond)), len(self.lambda_grid)))
        if add_gas_absorption:
            absorption_coeff += self._get_gas_absorption(abundances, P_cond, T_cond, zero_opacities=zero_opacities)
        if add_H_minus_absorption:
            absorption_coeff += self._get_H_minus_absorption(abundances, P_cond, T_cond)
        if add_scattering:
            if ri is not None:
                if scattering_factor != 1 or scattering_slope != 4:
                    raise ValueError("Cannot use both parametric and Mie scattering at the same time")
                
                absorption_coeff += self._get_mie_scattering_absorption(
                    P_cond, T_cond, ri, part_size,
                    frac_scale_height, number_density, sigma=part_size_std)
                absorption_coeff += self._get_scattering_absorption(
                    abundances, P_cond, T_cond)
                
            else:
                absorption_coeff += self._get_scattering_absorption(abundances,
                P_cond, T_cond, scattering_factor, scattering_slope,
                scattering_ref_wavelength)

        if add_collisional_absorption:
            absorption_coeff += self._get_collisional_absorption(
                abundances, P_cond, T_cond)

        # Cross sections vary less than absorption coefficients by pressure
        # and temperature, so interpolation should be done with cross sections
        cross_secs = absorption_coeff / (self.P_grid[P_cond][xp.newaxis, :, xp.newaxis] / k_B / self.T_grid[T_cond][:, xp.newaxis, xp.newaxis])
        cross_secs[cross_secs < min_cross_sec] = min_cross_sec
        
        if len(self.T_grid[T_cond]) == 1:
            cross_secs_atm = xp.exp(interp1d(xp.log(P_profile), xp.log(self.P_grid[P_cond]), xp.log(cross_secs[0])))
        else:            
            ln_cross = regular_grid_interp(
                1.0/self.T_grid[T_cond][::-1],
                xp.log(self.P_grid[P_cond]),
                xp.log(cross_secs[::-1]),
                1.0 / T_profile,
                xp.log(P_profile))
         
            cross_secs_atm = xp.exp(ln_cross)

        absorption_coeff_atm = cross_secs_atm * (P_profile / k_B / T_profile)[:, xp.newaxis]
        output_dict = {"absorption_coeff_atm": absorption_coeff_atm,
                       "radii": radii,
                       "dr": dr,
                       "P_profile": P_profile,
                       "T_profile": T_profile,
                       "mu_profile": mu_profile,
                       "atm_abundances": atm_abundances}
        
        return output_dict
