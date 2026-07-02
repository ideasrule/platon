import math

import numpy as np

from . import _forward_model as fm
from ._forward_model import ForwardConfig, ForwardInputs
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from .errors import AtmosphereError
from ._atmosphere_solver import AtmosphereSolver
from .params import NUM_LAYERS


def _pack_scalars(**kwargs):
    scalars = np.zeros(fm.SC_N_SCALARS, dtype=np.float64)
    for name, value in kwargs.items():
        scalars[getattr(fm, "SC_" + name.upper())] = value
    return scalars.astype(np.float32)


def _prepare_forward_inputs(atm, star_radius, planet_mass, planet_radius,
                            P_profile, T_profile, logZ, CO_ratio, CH4_mult,
                            gases, vmrs, add_gas_absorption,
                            add_H_minus_absorption, add_scattering,
                            scattering_factor, scattering_slope,
                            scattering_ref_wavelength,
                            add_collisional_absorption, cloudtop_pressure,
                            custom_abundances, T_star, T_spot, spot_cov_frac,
                            ri, frac_scale_height, number_density, part_size,
                            part_size_std, P_quench, min_abundance,
                            min_cross_sec, zero_opacities, stellar_blackbody,
                            bot_pressure, n_t_rows, surface_pressure=np.inf,
                            a_over_Rs=0.0, surface_temp=None, redist=0.0):
    """Host-side preparation shared by the transit and eclipse calculators.
    Returns (config kwargs dict, ForwardInputs, host bookkeeping dict)."""
    # bot_pressure is min(cloudtop_pressure, surface_pressure): the deepest
    # level light can reach, which is what must lie within the pressure grid
    atm._validate_params(T_profile, logZ, CO_ratio, bot_pressure)

    P_profile = np.asarray(P_profile, dtype=np.float64)
    T_profile = np.asarray(T_profile, dtype=np.float64)
    if not np.all(np.diff(P_profile) > 0):
        raise ValueError(
            "P_profile must be monotonically increasing in pressure")

    # Abundance mode
    vmrs_arr = None
    custom_log_abund = None
    gas_master_idx = ()
    if custom_abundances is None and logZ is not None and CO_ratio is not None:
        abund_mode = "eq"
        active_species = list(atm.abundance_getter.included_species)
    else:
        if logZ is not None or CO_ratio is not None:
            raise ValueError(
                "Must set logZ=None and CO_ratio=None to use custom_abundances")
        if custom_abundances is not None:
            if isinstance(custom_abundances, str):
                from .abundance_getter import AbundanceGetter
                custom_abundances = AbundanceGetter.from_file(custom_abundances)
            if not isinstance(custom_abundances, dict):
                raise ValueError("Unrecognized format for custom_abundances")
            abund_mode = "custom"
            custom_log_abund = atm.custom_abundances_to_log_master(
                custom_abundances)
            active_species = list(custom_abundances.keys())
        elif vmrs is not None and gases is not None:
            abund_mode = "vmr"
            for gas in gases:
                if gas not in atm.master_index:
                    raise ValueError("Unknown gas: {}".format(gas))
            gas_master_idx = tuple(int(atm.master_index[g]) for g in gases)
            vmrs_arr = np.asarray(vmrs, dtype=np.float32)
            active_species = list(gases)
        else:
            raise ValueError("Unrecognized format for custom_abundances")

    if add_H_minus_absorption and \
       ("el" not in active_species or "H" not in active_species):
        raise ValueError(
            "add_H_minus_absorption requires 'el' and 'H' abundances, which "
            "are missing from the provided gases/custom_abundances")

    # Mie scattering
    eff_xsec = None
    mie_ref_P = 1.0
    use_mie = ri is not None
    if use_mie:
        if scattering_factor != 1 or scattering_slope != 4:
            raise ValueError(
                "Cannot use both parametric and Mie scattering at the same time")
        eff_xsec = atm.get_mie_eff_cross_section(
            ri, part_size, sigma=part_size_std).astype(np.float32)
        mie_ref_P = atm.get_mie_ref_pressure(P_profile, bot_pressure)

    opac_mask = np.ones(len(atm.raw["opac_names"]), dtype=np.float32)
    for name in zero_opacities:
        if name in atm.raw["opac_names"]:
            opac_mask[atm.raw["opac_names"].index(name)] = 0

    n_above, shell_mask = atm.get_above_info(P_profile, bot_pressure)
    T_quench = atm.get_quench_T(P_profile, T_profile, P_quench)

    if T_spot is None:
        T_spot = T_star
    if spot_cov_frac is None:
        spot_cov_frac = 0.0

    scalars = _pack_scalars(
        rs=star_radius, mp=planet_mass, rp=planet_radius,
        logz=0.0 if logZ is None else logZ,
        co=0.0 if CO_ratio is None else CO_ratio,
        log_ch4=math.log10(max(CH4_mult, 1e-99)),
        scat_factor=scattering_factor, scat_slope=scattering_slope,
        scat_ref_um=scattering_ref_wavelength * 1e6,
        cloudtop=cloudtop_pressure,
        t_quench=T_quench, p_quench=P_quench,
        log10_p_quench=math.log10(max(P_quench, 1e-99)),
        t_star=0.0 if T_star is None else T_star,
        t_spot=0.0 if T_spot is None else T_spot,
        spot_frac=spot_cov_frac,
        fsh=frac_scale_height, num_den=number_density,
        ln_min_xsec=math.log(min_cross_sec),
        log_min_abund=math.log10(min_abundance),
        ref_pressure=atm.ref_pressure,
        t_star_hydro=Teff_sun if T_star is None else T_star,
        mie_ref_p=mie_ref_P,
        surface_p=surface_pressure,
        a_over_rs=a_over_Rs,
        surface_temp=0.0 if surface_temp is None else surface_temp,
        redist=redist,
    )

    if n_t_rows == 2:
        t0 = atm.get_t0(T_profile)
    else:
        t0 = 0
    idx_bot = int(shell_mask.sum()) - 1
    ints = np.array([t0, n_above - 1, idx_bot], dtype=np.int32)

    config_kwargs = dict(
        n_master=len(atm.master_names),
        n_t_rows=n_t_rows,
        abund_mode=abund_mode,
        gas_master_idx=gas_master_idx,
        ch4_idx=int(atm.master_index.get("CH4", -1)),
        el_idx=int(atm.master_index.get("el", 0)),
        h_idx=int(atm.master_index.get("H", 0)),
        add_gas=bool(add_gas_absorption),
        add_hminus=bool(add_H_minus_absorption),
        add_scattering=bool(add_scattering),
        add_collisional=bool(add_collisional_absorption),
        use_mie=use_mie and add_scattering,
        has_t_star=T_star is not None,
        blackbody=bool(stellar_blackbody),
        has_bins=atm.wavelength_bins is not None,
    )

    inputs = ForwardInputs(
        scalars=scalars, ints=ints,
        T_profile=T_profile.astype(np.float32),
        P_profile=P_profile.astype(np.float32),
        shell_mask=shell_mask, opac_mask=opac_mask,
        vmrs=vmrs_arr, custom_log_abund=custom_log_abund, eff_xsec=eff_xsec)

    host = dict(n_above=n_above, active_species=active_species,
                P_profile=P_profile, T_profile=T_profile)
    return config_kwargs, inputs, host


def _atm_info_dict(atm, out, host):
    """Build the backward-compatible full_output info dict entries shared by
    both calculators (arrays truncated to the above-cloud region)."""
    n = host["n_above"]
    atm_out = out.atm
    atm_abund = np.array(atm_out.atm_abund)      # (N, M)
    abundances = {}
    for name in host["active_species"]:
        idx = atm.master_index[name]
        abundances[name] = atm_abund[:n, idx]
    return dict(
        absorption_coeff_atm=np.array(atm_out.absorption_coeff_atm)[:n],
        radii=np.array(atm_out.radii)[:n],
        dr=np.array(atm_out.dr)[:n - 1],
        P_profile=np.array(host["P_profile"])[:n],
        T_profile=np.array(host["T_profile"])[:n],
        mu_profile=np.array(atm_out.mu_profile),
        atm_abundances=abundances,
    )


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

    def compute_depths(self, star_radius, planet_mass, planet_radius,
                       temperature, logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       full_output=False, min_abundance=1e-99,
                       min_cross_sec=1e-99, stellar_blackbody=False,
                       zero_opacities=[]):
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
        if custom_P_profile is not None:
            if custom_T_profile is None or len(
                    custom_P_profile) != len(custom_T_profile):
                raise ValueError("Must specify both custom_T_profile and "
                                 "custom_P_profile, and the two must have the"
                                 " same length")
            if temperature is not None:
                raise ValueError(
                    "Cannot specify both temperature and custom T profile")
            P_profile = np.asarray(custom_P_profile, dtype=np.float64)
            T_profile = np.asarray(custom_T_profile, dtype=np.float64)
        else:
            P_profile = np.logspace(
                np.log10(self.atm.P_grid[0]),
                np.log10(self.atm.P_grid[-1]),
                NUM_LAYERS)
            T_profile = np.ones(len(P_profile)) * temperature

        n_t_rows = 2 if T_profile.max() == T_profile.min() else self.atm.N_T

        config_kwargs, inputs, host = _prepare_forward_inputs(
            self.atm, star_radius, planet_mass, planet_radius,
            P_profile, T_profile, logZ, CO_ratio, CH4_mult, gases, vmrs,
            add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
            number_density, part_size, part_size_std, P_quench,
            min_abundance, min_cross_sec, zero_opacities, stellar_blackbody,
            bot_pressure=cloudtop_pressure, n_t_rows=n_t_rows)

        cfg = ForwardConfig(**config_kwargs)
        out = fm.transit_core(cfg, self.atm.device_data(), inputs)

        if bool(out.atm.unbound):
            raise AtmosphereError("Atmosphere unbound: height > hill radius")

        binned_depths = np.array(out.binned_depths, dtype=np.float64)
        if self.atm.wavelength_bins is None:
            binned_wavelengths = np.array(self.atm.lambda_grid)
        else:
            binned_wavelengths = np.array(
                self.atm._bin_info["transit_wavelengths"])

        if not full_output:
            return binned_wavelengths, binned_depths, None

        n = host["n_above"]
        atm_info = _atm_info_dict(self.atm, out, host)
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
                 for (l, r) in self.atm._bin_info["transit_ranges"]])

        return binned_wavelengths, binned_depths, atm_info
