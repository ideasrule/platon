import numpy as np
from scipy.special import expn

from .constants import h, c, k_B, G
from .params import NUM_LAYERS, MIN_P, MAX_P


def _default_pressures():
    return np.logspace(np.log10(MIN_P), np.log10(MAX_P), NUM_LAYERS)


class Profile:
    """A temperature/pressure profile.

    Create one with Profile(pressures, temperatures) to use arbitrary arrays
    directly, or with one of the class-method constructors, which compute
    the temperatures on the default pressure grid::

        Profile.isothermal(1000)
        Profile.parametric(T0, P1, alpha1, alpha2, P3, T3)
        Profile.guillot(T_irr, log_gamma, log_k_th, T_int, Mp, Rp)
        Profile.radiative_solution(T_star, Rs, a, Mp, Rp, beta, ...)
        Profile.from_opacity(T_irr, info_dict)
        Profile.from_arrays(P_profile, T_profile)
        Profile.from_params_dict(profile_type, params_dict)

    T/P profiles are computed on the host in numpy (float64); they are tiny
    arrays, and several profile types need scipy.special.expn."""

    def __init__(self, pressures, temperatures):
        """Profile with the given temperatures (K) at the given pressures
        (Pa), used as-is with no interpolation."""
        self.pressures = np.asarray(pressures, dtype=np.float64)
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        if self.pressures.shape != self.temperatures.shape:
            raise ValueError(
                "pressures and temperatures must have the same shape")
        self.profile_type = None
        self.profile_params = {}

    @classmethod
    def _parameterized(cls, temperatures, profile_type, profile_params):
        profile = cls(_default_pressures(), temperatures)
        profile.profile_type = profile_type
        profile.profile_params = profile_params
        return profile

    def get_temperatures(self):
        return np.array(self.temperatures)

    def get_pressures(self):
        return np.array(self.pressures)

    @classmethod
    def from_params_dict(cls, profile_type, params_dict, suffix=""):
        """Creates the profile from parameters named in params_dict.  If
        suffix is given (e.g. "_transit"), parameters with the suffixed name
        (e.g. T0_transit) override the unsuffixed ones (e.g. T0), allowing
        separate profiles to coexist in one params_dict."""
        if suffix:
            params_dict = dict(params_dict)
            for name, value in list(params_dict.items()):
                if name.endswith(suffix) and value is not None:
                    params_dict[name[:-len(suffix)]] = value
        if profile_type == "isothermal":
            return cls.isothermal(params_dict["T"])
        elif profile_type == "parametric":
            return cls.parametric(
                params_dict["T0"], 10**params_dict["log_P1"],
                params_dict["alpha1"], params_dict["alpha2"],
                10**params_dict["log_P3"], params_dict["T3"])
        elif profile_type == "radiative_solution":
            return cls.radiative_solution(**params_dict)
        elif profile_type == "guillot":
            return cls.guillot(
                params_dict["T_irr"], params_dict["log_gamma"],
                params_dict["log_k_th"], params_dict.get("T_int", 100),
                params_dict["Mp"], params_dict["Rp"])
        else:
            raise ValueError("Unknown profile type: {}".format(profile_type))

    @classmethod
    def from_arrays(cls, P_profile, T_profile):
        """Interpolates the given profile (in log P) onto the default pressure
        grid.  To use the arrays as-is, call Profile(pressures, temperatures)
        instead."""
        P_profile = np.asarray(P_profile)
        T_profile = np.asarray(T_profile)
        temperatures = np.interp(np.log10(_default_pressures()),
                                 np.log10(P_profile), T_profile)
        return cls._parameterized(temperatures, "arrays", {})

    @classmethod
    def isothermal(cls, T_day):
        temperatures = np.ones(NUM_LAYERS) * T_day
        return cls._parameterized(temperatures, "isothermal", {"T": T_day})

    @classmethod
    def parametric(cls, T0, P1, alpha1, alpha2, P3, T3):
        '''Parametric model from https://arxiv.org/pdf/0910.1347.pdf'''
        P = _default_pressures()
        P0 = np.min(P)

        ln_P2 = alpha2**2 * (T0 + np.log(P1 / P0)**2 / alpha1**2 - T3) - \
            np.log(P1)**2 + np.log(P3)**2
        ln_P2 /= 2 * np.log(P3 / P1)
        P2 = np.exp(ln_P2)
        T2 = T3 - np.log(P3 / P2)**2 / alpha2**2

        with np.errstate(divide="ignore", invalid="ignore"):
            temperatures = np.where(
                P < P1, T0 + np.log(P / P0)**2 / alpha1**2,
                np.where(P < P3, T2 + np.log(P / P2)**2 / alpha2**2, T3))
        return cls._parameterized(temperatures, "parametric", dict(
            T0=T0, P1=P1, alpha1=alpha1, alpha2=alpha2, P3=P3, T3=T3))

    @classmethod
    def from_opacity(cls, T_irr, info_dict, visible_cutoff=0.8e-6,
                     T_int=100):
        wavelengths = np.asarray(info_dict["unbinned_wavelengths"],
                                 dtype=np.float64)
        d_lambda = np.diff(wavelengths)
        d_lambda = np.append(d_lambda[0], d_lambda)

        # Convert stellar spectrum from photons/time to energy/time
        stellar_spectrum = np.asarray(info_dict["stellar_spectrum"],
                                      dtype=np.float64) * h * c / wavelengths

        # Convert planetary spectrum from energy/time/wavelength to energy/time
        planet_spectrum = np.asarray(info_dict["planet_spectrum"],
                                     dtype=np.float64) * d_lambda
        absorption_coeffs = np.asarray(info_dict["absorption_coeff_atm"],
                                       dtype=np.float64)
        radii = np.asarray(info_dict["radii"], dtype=np.float64)

        # Equation 49 here: https://arxiv.org/pdf/1006.4702.pdf
        visible = wavelengths < visible_cutoff
        thermal = wavelengths >= visible_cutoff
        n = np.asarray(info_dict["P_profile"], dtype=np.float64) / k_B / \
            np.asarray(info_dict["T_profile"], dtype=np.float64)
        intermediate_n = (n[0:-1] + n[1:]) / 2.0
        sigmas = absorption_coeffs / n[:, np.newaxis]
        sigma_v = np.median(np.average(sigmas[:, visible], axis=1,
                                       weights=stellar_spectrum[visible]))
        sigma_th = np.median(np.average(sigmas[:, thermal], axis=1,
                                        weights=planet_spectrum[thermal]))

        gamma = sigma_v / sigma_th

        dr = -np.diff(radii)
        d_taus = sigma_th * intermediate_n * dr
        taus = np.cumsum(d_taus)

        e2 = expn(2, gamma * taus)
        T4 = 3.0 / 4 * T_int**4 * (2.0 / 3 + taus) + \
            3.0 / 4 * T_irr**4 * (2.0 / 3 + 2.0 / 3 / gamma *
                                  (1 + (gamma * taus / 2 - 1) * np.exp(-gamma * taus)) +
                                  2.0 * gamma / 3 * (1 - taus**2 / 2) * e2)
        T = T4 ** 0.25
        return cls._parameterized(np.append(T[0], T), "opacity",
                                  dict(T_irr=T_irr, T_int=T_int))

    @classmethod
    def guillot(cls, T_irr, log_gamma, log_k_th, T_int, Mp, Rp):
        """The one-visible-channel profile from Guillot (2010).  log_k_th is
        log10 of the thermal opacity in m^2/kg; all other inputs are SI."""
        gamma = 10**log_gamma
        kappa_th = 10**log_k_th
        tau = _default_pressures() * kappa_th / (G * Mp / Rp**2)
        incoming = 2 / 3 + 2 / (3 * gamma) * (
            1 + (gamma * tau / 2 - 1) * np.exp(-gamma * tau))
        incoming += 2 * gamma / 3 * (1 - tau**2 / 2) * expn(2, gamma * tau)
        T4 = 3 / 4 * T_int**4 * (tau + 2 / 3) + \
            3 / 4 * T_irr**4 * incoming
        return cls._parameterized(T4**0.25, "guillot", dict(
            T_irr=T_irr, log_gamma=log_gamma, log_k_th=log_k_th,
            T_int=T_int, Mp=Mp, Rp=Rp))

    @classmethod
    def radiative_solution(cls, T_star, Rs, a, Mp, Rp, beta,
                           log_k_th, log_gamma, log_gamma2=None,
                           alpha=0, T_int=100, **ignored_kwargs):
        '''From Line et al. 2013: http://adsabs.harvard.edu/abs/2013ApJ...775..137L, Equation 13 - 16.
        log_k_th is log10 of the thermal opacity in m^2/kg; all other inputs
        are SI.'''

        k_th = 10.0**log_k_th
        gamma = 10.0**log_gamma
        gamma2 = 10.0**log_gamma2 if log_gamma2 is not None else None

        g = G * Mp / Rp**2
        T_eq = beta * np.sqrt(Rs / (2 * a)) * T_star
        taus = k_th * _default_pressures() / g

        def incoming_stream_contribution(gamma):
            return 3.0 / 4 * T_eq**4 * \
                (2.0 / 3 + 2.0 / 3 / gamma *
                 (1 + (gamma * taus / 2 - 1) * np.exp(-gamma * taus)) +
                 2.0 * gamma / 3 * (1 - taus**2 / 2) * expn(2, gamma * taus))

        e1 = incoming_stream_contribution(gamma)
        T4 = 3.0 / 4 * T_int**4 * (2.0 / 3 + taus) + (1 - alpha) * e1

        if gamma2 is not None:
            e2 = incoming_stream_contribution(gamma2)
            T4 += alpha * e2
        return cls._parameterized(T4 ** 0.25, "radiative_solution", dict(
            T_star=T_star, Rs=Rs, a=a, Mp=Mp, Rp=Rp, beta=beta,
            log_k_th=log_k_th, log_gamma=log_gamma,
            log_gamma2=log_gamma2, alpha=alpha, T_int=T_int))
