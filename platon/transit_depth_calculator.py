import numpy as np

from . import _forward_model as fm
from ._forward_prep import prepare_forward_inputs, atm_info_dict
from .errors import AtmosphereError
from ._atmosphere_solver import AtmosphereSolver
from .TP_profile import Profile


class TransitDepthCalculator:
    def __init__(self, include_condensation=True, ref_pressure=1e5,
                 method='xsec',
                 include_opacities=["CH4", "CO2", "CO", "H2O", "H2S", "HCN",
                                    "K", "Na", "NH3", "SO2", "TiO", "VO"],
                 downsample=1):
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
            "xsec" for opacity sampling (correlated-k is no longer supported)
        '''
        self.atm = AtmosphereSolver(include_condensation, ref_pressure,
                                    method, include_opacities, downsample)

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
        """
        self.atm.change_wavelength_bins(bins)

    def _validate_params(self, T, logZ, CO_ratio, cloudtop_pressure):
        self.atm._validate_params(T, logZ, CO_ratio, cloudtop_pressure)

    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       full_output=False, min_abundance=1e-99,
                       min_cross_sec=1e-99, stellar_blackbody=False,
                       zero_opacities=[]):
        '''
        Computes transit depths at a range of wavelengths.  To choose bins,
        call change_wavelength_bins().

        Parameters
        ----------
        t_p_profile : Profile
            A Profile object from TP_profile, describing the (potentially
            non-isothermal) T/P profile of the atmosphere.  For an
            isothermal atmosphere, use Profile.set_isothermal.
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
            Use np.inf for a cloudless atmosphere.
        custom_abundances : str or dict of np.ndarray, optional
            If specified, overrides `logZ` and `CO_ratio`.  The recommended
            format is a dictionary mapping species names to abundance
            profiles: 1D arrays with one element per atmospheric layer
            (ordered by increasing pressure, matching t_p_profile.pressures),
            so that custom_abundances['Na'][i]
            is the fractional number abundance of Na at the i-th layer.
            Alternatively, can specify a filename, in which case the
            abundances are read from a file in the format of the EOS/ files
            (identical to ExoTransmit's EOS files); or, in the legacy
            format, a dictionary mapping species names to (N_T, N_P) arrays
            on the temperature/pressure grid, so that
            custom_abundances['Na'][3,4] would mean the fractional number
            abundance of Na at a temperature of self.T_grid[3] and pressure
            of self.P_grid[4].  Grid-format abundances are interpolated
            onto the atmospheric layers.
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
            Whether to use a blackbody for the stellar spectrum instead of a
            PHOENIX model
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
        if not isinstance(t_p_profile, Profile):
            raise TypeError("t_p_profile must be a Profile object from "
                            "platon.TP_profile; for an isothermal "
                            "atmosphere, use Profile.set_isothermal")
        T_profile = np.asarray(t_p_profile.temperatures, dtype=np.float64)
        P_profile = np.asarray(t_p_profile.pressures, dtype=np.float64)

        cfg, inputs, host = prepare_forward_inputs(
            self.atm, star_radius=star_radius, planet_mass=planet_mass,
            planet_radius=planet_radius, P_profile=P_profile,
            T_profile=T_profile, logZ=logZ, CO_ratio=CO_ratio,
            CH4_mult=CH4_mult, gases=gases, vmrs=vmrs,
            add_gas_absorption=add_gas_absorption,
            add_H_minus_absorption=add_H_minus_absorption,
            add_scattering=add_scattering,
            scattering_factor=scattering_factor,
            scattering_slope=scattering_slope,
            scattering_ref_wavelength=scattering_ref_wavelength,
            add_collisional_absorption=add_collisional_absorption,
            cloudtop_pressure=cloudtop_pressure,
            custom_abundances=custom_abundances, T_star=T_star, T_spot=T_spot,
            spot_cov_frac=spot_cov_frac, ri=ri,
            frac_scale_height=frac_scale_height,
            number_density=number_density, part_size=part_size,
            part_size_std=part_size_std, P_quench=P_quench,
            min_abundance=min_abundance, min_cross_sec=min_cross_sec,
            zero_opacities=zero_opacities,
            stellar_blackbody=stellar_blackbody,
            bot_pressure=cloudtop_pressure)

        if full_output:
            out = fm.transit_core(cfg, self.atm.device_data(), inputs)
            binned, unbound = out.binned_depths, bool(out.atm.unbound)
        else:
            binned, unbound = fm.split_transit_result(np.asarray(
                fm.transit_depths_core(cfg, self.atm.device_data(), inputs)))

        if unbound:
            raise AtmosphereError("Atmosphere unbound: height > hill radius")

        binned_depths = np.array(binned, dtype=np.float64)
        if self.atm.wavelength_bins is None:
            binned_wavelengths = np.array(self.atm.lambda_grid)
        else:
            binned_wavelengths = np.array(
                self.atm._bin_info["bin_wavelengths"])

        if not full_output:
            return binned_wavelengths, binned_depths, None

        n = host["n_above"]
        atm_info = atm_info_dict(self.atm, out, host)
        stellar = np.array(out.stellar_spectrum, dtype=np.float64)
        corr = np.array(out.correction_factors, dtype=np.float64)
        depths_uncorr = np.array(out.depths, dtype=np.float64)
        atm_info["tau_los"] = np.array(out.tau_los)[:, :n - 1]
        atm_info["contrib"] = np.array(out.absorption_fraction)[:, :n - 1]
        atm_info["unbinned_wavelengths"] = np.array(self.atm.lambda_grid)
        atm_info["unbinned_stellar_spectrum"] = stellar
        atm_info["unbinned_correction_factors"] = corr
        if self.atm.wavelength_bins is None:
            atm_info["unbinned_depths"] = depths_uncorr * corr
            atm_info["binned_stellar_spectrum"] = stellar
        else:
            atm_info["unbinned_depths"] = depths_uncorr
            atm_info["binned_stellar_spectrum"] = np.array(
                [np.median(stellar[l:r])
                 for (l, r) in self.atm._bin_info["bin_ranges"]])

        return binned_wavelengths, binned_depths, atm_info
