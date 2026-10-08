import matplotlib.pyplot as plt
import scipy
import numpy as np

from . import _cupy_numpy as xp
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
from .params import NUM_LAYERS


class TransitDepthCalculator:
    def __init__(self, include_condensation=True, ref_pressure=1e5, method='xsec',include_opacities=["NH3", "H2O", "CO", "CH4", "H2S", "HCN", "CS2", "DMS", "CH3SH"], downsample=1):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        ref_pressure : float
            The planetary radius is defined as the radius at this pressure
        method : string
            "xsec" for opacity sampling, "ktables" for correlated k
        startag : string
            custom stellar spectra pkl file
        '''
        self.atm = AtmosphereSolver(include_condensation, ref_pressure, method, include_opacities, downsample)

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
                                     spot_cov_frac, blackbody=False, n_gauss=10):
        depths = xp.cpu(depths)
        unbinned_lambdas = xp.cpu(self.atm.lambda_grid)
        stellar_spectrum, correction_factors = self.atm.get_stellar_spectrum(
            T_star, T_spot, spot_cov_frac, blackbody)
        stellar_spectrum = xp.cpu(stellar_spectrum)
        correction_factors = xp.cpu(correction_factors)
        
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
            return xp.array(intermediate_lambdas),\
                xp.array(intermediate_depths * intermediate_correction_factors),\
                xp.array(intermediate_stellar_spectrum),\
                xp.array(intermediate_lambdas),\
                xp.array(intermediate_depths * intermediate_correction_factors),\
                xp.array(intermediate_stellar_spectrum),\
                xp.array(intermediate_correction_factors)
                        
        binned_wavelengths = []
        binned_depths = []
        binned_stellar_spectrum = []
        
        for (start, end) in xp.cpu(self.atm.wavelength_bins):
            l = np.searchsorted(intermediate_lambdas, start)
            r = np.searchsorted(intermediate_lambdas, end)
            
            binned_wavelengths.append(np.mean(intermediate_lambdas[l:r]))
            binned_depth = np.average(intermediate_depths[l:r] * intermediate_correction_factors[l:r],
                                      weights=intermediate_stellar_spectrum[l:r])
            binned_depths.append(binned_depth)
            binned_stellar_spectrum.append(np.median(intermediate_stellar_spectrum[l:r]))

        return xp.array(binned_wavelengths), xp.array(binned_depths), xp.array(binned_stellar_spectrum), xp.array(intermediate_lambdas), xp.array(intermediate_depths), xp.array(intermediate_stellar_spectrum), xp.array(intermediate_correction_factors)

    def _validate_params(self, T, logZ, CO_ratio, cloudtop_pressure):
        self.atm._validate_params(T, logZ, CO_ratio, cloudtop_pressure)

    @staticmethod
    def _to_cpu_recursive(value):
        if isinstance(value, dict):
            return {
                key: TransitDepthCalculator._to_cpu_recursive(subvalue)
                for key, subvalue in value.items()
            }
        return xp.cpu(value)

    def _resolve_profile_arrays(self, t_p_profile, custom_T_profile=None,
                                custom_P_profile=None):
        if custom_P_profile is not None:
            if custom_T_profile is None or len(custom_P_profile) != len(custom_T_profile):
                raise ValueError("Must specify both custom_T_profile and "
                                 "custom_P_profile, and the two must have the"
                                 " same length")
            return custom_P_profile, custom_T_profile

        return t_p_profile.pressures, t_p_profile.temperatures

    def _compute_unbinned_depths(self, t_p_profile, star_radius, planet_mass,
                                 planet_radius, logZ=0, CO_ratio=0.53,
                                 CH4_mult=1, gases=None, vmrs=None,
                                 log_SO2=None, log_CH4=None, log_TiO=None,
                                 log_VO=None, log_CS2=None, add_gas_absorption=True,
                                 add_H_minus_absorption=False,
                                 add_scattering=True, scattering_factor=1,
                                 scattering_slope=4,
                                 scattering_ref_wavelength=1e-6,
                                 add_collisional_absorption=True,
                                 cloudtop_pressure=xp.inf,
                                 custom_abundances=None,
                                 custom_T_profile=None, custom_P_profile=None,
                                 T_star=None, T_spot=None, spot_cov_frac=None,
                                 ri=None, frac_scale_height=1,
                                 number_density=0, part_size=1e-6,
                                 part_size_std=0.5, P_quench=1e-99,
                                 T_quench=None, quench_species=None,
                                 min_abundance=1e-99, min_cross_sec=1e-99,
                                 zero_opacities=[]):
        P_profile, T_profile = self._resolve_profile_arrays(
            t_p_profile, custom_T_profile, custom_P_profile)

        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, CH4_mult, gases, vmrs, add_gas_absorption, add_H_minus_absorption,
            add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
            number_density, part_size, part_size_std, P_quench, min_abundance, min_cross_sec, zero_opacities,
            log_CS2=log_CS2)

        radii = atm_info["radii"]
        dr = atm_info["dr"]
        tau_los = get_line_of_sight_tau(atm_info["absorption_coeff_atm"], radii)
        absorption_fraction = 1 - xp.exp(-tau_los)
        transit_depths = ((radii.min() / star_radius) ** 2
                          + 2 / star_radius ** 2
                          * absorption_fraction.dot(radii[1:] * dr))
        return transit_depths, atm_info, tau_los, absorption_fraction

    def _finalize_depths(self, transit_depths, T_star, T_spot, spot_cov_frac,
                         stellar_blackbody=False, full_output=False,
                         atm_info=None, tau_los=None,
                         absorption_fraction=None, extra_info=None):
        # For correlated-k: transit_depths has n_gauss points for every
        # wavelength; unbinned_depths has 1 point for every wavelength.
        (binned_wavelengths, binned_depths, binned_stellar_spectrum,
         unbinned_wavelengths, unbinned_depths, unbinned_stellar_spectrum,
         unbinned_correction_factors) = self._get_binned_corrected_depths(
             transit_depths, T_star, T_spot, spot_cov_frac, stellar_blackbody)

        if not full_output:
            return xp.cpu(binned_wavelengths), xp.cpu(binned_depths), None

        info_dict = {} if atm_info is None else dict(atm_info)
        if tau_los is not None:
            info_dict["tau_los"] = tau_los
        info_dict["binned_stellar_spectrum"] = binned_stellar_spectrum
        info_dict["unbinned_wavelengths"] = unbinned_wavelengths
        info_dict["unbinned_depths"] = unbinned_depths
        info_dict["unbinned_stellar_spectrum"] = unbinned_stellar_spectrum
        info_dict["unbinned_correction_factors"] = unbinned_correction_factors
        if absorption_fraction is not None:
            info_dict["contrib"] = absorption_fraction
        if extra_info is not None:
            info_dict.update(extra_info)

        return (xp.cpu(binned_wavelengths), xp.cpu(binned_depths),
                self._to_cpu_recursive(info_dict))

    def compute_depths(self, t_p_profile, star_radius, planet_mass, planet_radius,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       log_SO2=None, log_CH4=None, log_TiO=None,
                       log_VO=None, log_CS2=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       T_quench=None, quench_species=None,
                       full_output=False, min_abundance=1e-99, min_cross_sec=1e-99, stellar_blackbody=False, zero_opacities=[]):
        '''
        Computes transit depths at a range of wavelengths, assuming an
        isothermal atmosphere.  To choose bins, call change_wavelength_bins().

        Parameters
        ----------
        t_p_profile : float
            Profile object
        star_radius : float
            Radius of the star
        planet_mass : float
            Mass of the planet, in kg
        planet_radius : float
            Radius of the planet at 100,000 Pa. Must be in metres.
        logZ : float
            Base-10 logarithm of the metallicity, in solar units
        CO_ratio : float, optional
            C/O atomic ratio in the atmosphere.  The solar value is 0.53.
        CH4_mult : float
            Multiple applied to equilibrium CH4 abundance, for methane depletion
        add_gas_absorption: float, optional
            Whether gas absorption is accounted for
        add_H_minus_absorption: float, optional
            Whether H- bound-free and free-free absorption is added in
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
            Use xp.inf for a cloudless atmosphere.
        custom_abundances : str or dict of xp.ndarray, optional
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
        P_quench : float, optional
            Quench pressure in Pa.
        stellar_blackbody : bool, optional
            Whether to use a PHOENIX model for the stellar spectrum, or a blackbody
        zero_opacities : list of strings                                                                                                                                                                   
            List of molecules to zero opacities for
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
        transit_depths, atm_info, tau_los, absorption_fraction = (
            self._compute_unbinned_depths(
                t_p_profile, star_radius, planet_mass, planet_radius,
                logZ, CO_ratio, CH4_mult, gases, vmrs,
                log_SO2, log_CH4, log_TiO, log_VO, log_CS2,
                add_gas_absorption, add_H_minus_absorption,
                add_scattering, scattering_factor, scattering_slope,
                scattering_ref_wavelength, add_collisional_absorption,
                cloudtop_pressure, custom_abundances,
                custom_T_profile, custom_P_profile,
                T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
                number_density, part_size, part_size_std, P_quench,
                T_quench, quench_species, min_abundance, min_cross_sec,
                zero_opacities))

        return self._finalize_depths(
            transit_depths, T_star, T_spot, spot_cov_frac,
            stellar_blackbody=stellar_blackbody, full_output=full_output,
            atm_info=atm_info, tau_los=tau_los,
            absorption_fraction=absorption_fraction)

    def compute_depths_patchy(self, t_p_profile, star_radius, planet_mass, planet_radius,
                       cloud_cov_frac,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       log_SO2=None, log_CH4=None, log_TiO=None,
                       log_VO=None, log_CS2=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       T_quench=None, quench_species=None,
                       full_output=False, min_abundance=1e-99, min_cross_sec=1e-99, stellar_blackbody=False, zero_opacities=[]):
        """
        Computes transit depths for a patchy cloud model.
        This method is an optimized version for patchy clouds. It first computes
        the clear-sky model to cache the gas absorption, then computes the cloudy
        model reusing the cached data, and finally combines them.
        """
        # Cache should only be active within this call, even if an exception occurs.
        self.atm.cache_gas_absorption = True
        self.atm.cache_scattering_base = True
        self.atm.cache_profile_quantities = True
        try:
            depths_clear, info_clear, tau_clear, contrib_clear = (
                self._compute_unbinned_depths(
                    t_p_profile, star_radius, planet_mass, planet_radius,
                    logZ, CO_ratio, CH4_mult, gases, vmrs,
                    log_SO2, log_CH4, log_TiO, log_VO, log_CS2,
                    add_gas_absorption, add_H_minus_absorption,
                    True, 1, 4, scattering_ref_wavelength,
                    add_collisional_absorption, xp.inf, custom_abundances,
                    custom_T_profile, custom_P_profile,
                    T_star, T_spot, spot_cov_frac,
                    None, 1, 0,
                    part_size, part_size_std, P_quench, T_quench,
                    quench_species, min_abundance, min_cross_sec,
                    zero_opacities))

            depths_cloudy, info_cloudy, tau_cloudy, contrib_cloudy = (
                self._compute_unbinned_depths(
                    t_p_profile, star_radius, planet_mass, planet_radius,
                    logZ, CO_ratio, CH4_mult, gases, vmrs,
                    log_SO2, log_CH4, log_TiO, log_VO, log_CS2,
                    add_gas_absorption, add_H_minus_absorption,
                    add_scattering, scattering_factor, scattering_slope,
                    scattering_ref_wavelength, add_collisional_absorption,
                    cloudtop_pressure, custom_abundances,
                    custom_T_profile, custom_P_profile,
                    T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
                    number_density, part_size, part_size_std, P_quench,
                    T_quench, quench_species, min_abundance, min_cross_sec,
                    zero_opacities))
        finally:
            self.atm.cache_gas_absorption = False
            self.atm.gas_absorption_cache = None
            self.atm.cache_scattering_base = False
            self.atm.scattering_base_cache = None
            self.atm.cache_profile_quantities = False
            self.atm.profile_quantities_cache = None

        final_depths = cloud_cov_frac * depths_cloudy + (1 - cloud_cov_frac) * depths_clear

        extra_info = None
        if full_output:
            extra_info = {
                "model_mode": "patchy",
                "cloud_cov_frac": cloud_cov_frac,
                "clear_unbinned_depths": depths_clear,
                "cloudy_unbinned_depths": depths_cloudy,
            }

        return self._finalize_depths(
            final_depths, T_star, T_spot, spot_cov_frac,
            stellar_blackbody=stellar_blackbody, full_output=full_output,
            atm_info=info_cloudy, tau_los=tau_cloudy,
            absorption_fraction=contrib_cloudy, extra_info=extra_info)

    def compute_depths_limb_asym(self, t_p_profile_morning, t_p_profile_evening,
                       star_radius, planet_mass, planet_radius,
                       f_limb=0.5,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       log_SO2=None, log_CH4=None, log_TiO=None,
                       log_VO=None, log_CS2=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       scattering_factor_evening=1,
                       scattering_slope_evening=4,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf,
                       cloudtop_pressure_evening=xp.inf,
                       custom_abundances=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       T_quench=None, quench_species=None,
                       full_output=False, min_abundance=1e-99, min_cross_sec=1e-99,
                       stellar_blackbody=False, zero_opacities=[]):
        """Compute transit depths for a 1.5D limb-asymmetric model.

        Morning limb: cloudy (cloudtop_pressure, scattering_factor,
        scattering_slope), uses t_p_profile_morning.
        Evening limb: uses t_p_profile_evening with its own cloud-top pressure
        (cloudtop_pressure_evening) and its own scattering parameters
        (scattering_factor_evening, scattering_slope_evening). Defaults match
        a Rayleigh-only clear limb.
        Final depths = f_limb * morning + (1 - f_limb) * evening.
        """
        self.atm.cache_gas_absorption = True
        self.atm.cache_scattering_base = True
        # Morning and evening limbs use different T/P profiles, so hydrostatic
        # structure and profile-interpolated abundances must be recomputed for
        # each call. Only the profile-independent opacity bases are safe to
        # reuse here.
        self.atm.cache_profile_quantities = False
        try:
            depths_evening, info_evening, tau_evening, contrib_evening = (
                self._compute_unbinned_depths(
                    t_p_profile_evening, star_radius, planet_mass, planet_radius,
                    logZ, CO_ratio, CH4_mult, gases, vmrs,
                    log_SO2, log_CH4, log_TiO, log_VO, log_CS2,
                    add_gas_absorption, add_H_minus_absorption,
                    True, scattering_factor_evening, scattering_slope_evening,
                    scattering_ref_wavelength, add_collisional_absorption,
                    cloudtop_pressure_evening, custom_abundances,
                    None, None,
                    T_star, T_spot, spot_cov_frac,
                    None, 1, 0,
                    part_size, part_size_std, P_quench, T_quench,
                    quench_species, min_abundance, min_cross_sec,
                    zero_opacities))

            depths_morning, info_morning, tau_morning, contrib_morning = (
                self._compute_unbinned_depths(
                    t_p_profile_morning, star_radius, planet_mass, planet_radius,
                    logZ, CO_ratio, CH4_mult, gases, vmrs,
                    log_SO2, log_CH4, log_TiO, log_VO, log_CS2,
                    add_gas_absorption, add_H_minus_absorption,
                    add_scattering, scattering_factor, scattering_slope,
                    scattering_ref_wavelength, add_collisional_absorption,
                    cloudtop_pressure, custom_abundances,
                    None, None,
                    T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
                    number_density, part_size, part_size_std, P_quench,
                    T_quench, quench_species, min_abundance, min_cross_sec,
                    zero_opacities))
        finally:
            self.atm.cache_gas_absorption = False
            self.atm.gas_absorption_cache = None
            self.atm.cache_scattering_base = False
            self.atm.scattering_base_cache = None
            self.atm.cache_profile_quantities = False
            self.atm.profile_quantities_cache = None

        # 1.5D linear blend: each sector's unbinned depth is already a full
        # transit depth (solid body + annular), so the angular-weighted
        # average of the two gives the correct terminator-integrated depth.
        # When delta_T = 0 and all other params match, this collapses back
        # to the 1D model exactly. Stellar contamination and wavelength
        # binning are applied once to the blended spectrum in _finalize.
        final_depths = f_limb * depths_morning + (1 - f_limb) * depths_evening

        extra_info = None
        if full_output:
            extra_info = {
                "model_mode": "limb_asym",
                "f_limb": f_limb,
                "morning_unbinned_depths": depths_morning,
                "evening_unbinned_depths": depths_evening,
                "morning_radii": info_morning["radii"],
                "evening_radii": info_evening["radii"],
                "morning_P_profile": info_morning["P_profile"],
                "evening_P_profile": info_evening["P_profile"],
                "morning_T_profile": info_morning["T_profile"],
                "evening_T_profile": info_evening["T_profile"],
                "morning_mu_profile": info_morning["mu_profile"],
                "evening_mu_profile": info_evening["mu_profile"],
                "morning_atm_abundances": info_morning["atm_abundances"],
                "evening_atm_abundances": info_evening["atm_abundances"],
            }

        return self._finalize_depths(
            final_depths, T_star, T_spot, spot_cov_frac,
            stellar_blackbody=stellar_blackbody, full_output=full_output,
            atm_info=info_morning, tau_los=tau_morning,
            absorption_fraction=contrib_morning, extra_info=extra_info)
