from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.special
import numpy as np

from pkg_resources import resource_filename

from .constants import h, c, k_B, AMU, G

class Profile:
    def __init__(self, num_profile_heights=250, min_P=1e-4, max_P=1e8):
        self.pressures = np.logspace(
                np.log10(min_P),
                np.log10(max_P),
                num_profile_heights)
        
    def set_from_params_dict(self, profile_type, params_dict):
        if profile_type == "isothermal":
            self.set_isothermal(params_dict["T"])
        elif profile_type == "parametric":
            self.set_parametric(
                params_dict["T0"], 10**params_dict["log_P1"],
                params_dict["alpha1"], params_dict["alpha2"],
                10**params_dict["log_P3"], params_dict["T3"])
        elif profile_type == "radiative_solution":
            self.set_from_radiative_solution(**params_dict)
        else:
            assert(False)
                                        
        
    def set_from_arrays(self, P_profile, T_profile):
        interpolator = interp1d(np.log10(P_profile), T_profile)
        self.temperatures = interpolator(np.log10(self.pressures))

    def set_isothermal(self, T_day):
        self.temperatures = np.ones(len(self.pressures)) * T_day

    def set_parametric(self, T0, P1, alpha1, alpha2, P3, T3):
        '''Parametric model from https://arxiv.org/pdf/0910.1347.pdf'''
        P0 = np.min(self.pressures)

        ln_P2 = alpha2**2*(T0+np.log(P1/P0)**2/alpha1**2 - T3) - np.log(P1)**2 + np.log(P3)**2
        ln_P2 /= 2 * np.log(P3/P1)
        P2 = np.exp(ln_P2)
        T2 = T3 - np.log(P3/P2)**2/alpha2**2

        self.temperatures = np.zeros(len(self.pressures))
        for i, P in enumerate(self.pressures):
            if P < P1:
                self.temperatures[i] = T0 + np.log(P/P0)**2 / alpha1**2
            elif P < P3:
                self.temperatures[i] = T2 + np.log(P/P2)**2 / alpha2**2
            else:
                self.temperatures[i] = T3
        return P2, T2

    def set_from_opacity(self, T_irr, info_dict, visible_cutoff=0.8e-6, T_int=100):
        wavelengths = info_dict["unbinned_wavelengths"]
        d_lambda = np.diff(wavelengths)
        d_lambda = np.append(d_lambda[0], d_lambda)

        # Convert stellar spectrum from photons/time to energy/time
        stellar_spectrum = info_dict["stellar_spectrum"] * h * c / wavelengths

        # Convert planetary spectrum from energy/time/wavelength to energy/time
        planet_spectrum = info_dict["planet_spectrum"] * d_lambda
        absorption_coeffs = info_dict["absorption_coeff_atm"]
        radii = info_dict["radii"]

        # Equation 49 here: https://arxiv.org/pdf/1006.4702.pdf
        visible = wavelengths < visible_cutoff
        thermal = wavelengths >= visible_cutoff
        n = info_dict["P_profile"]/k_B/info_dict["T_profile"]
        intermediate_n = (n[0:-1] + n[1:])/2.0
        sigmas = absorption_coeffs / n[:, np.newaxis]
        sigma_v = np.median(np.average(sigmas[:, visible], axis=1, weights=stellar_spectrum[visible]))
        sigma_th = np.median(np.average(sigmas[:, thermal], axis=1, weights=planet_spectrum[thermal]))

        gamma = sigma_v / sigma_th

        dr = -np.diff(radii)
        d_taus = sigma_th * intermediate_n * dr
        taus = np.cumsum(d_taus)

        e2 = scipy.special.expn(2, gamma*taus)
        T4 = 3.0/4 * T_int**4 * (2.0/3 + taus) + 3.0/4 * T_irr**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*np.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * e2)
        T = T4 ** 0.25
        self.temperatures = np.append(T[0], T)

    def set_from_radiative_solution(self, T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, log_gamma2=None, alpha=0, T_int=100, **ignored_kwargs):
        '''From Line et al. 2013: http://adsabs.harvard.edu/abs/2013ApJ...775..137L, Equation 13 - 16'''

        k_th = 10.0**log_k_th
        gamma = 10.0**log_gamma
        gamma2 = 10.0**log_gamma2
        
        g = G * Mp / Rp**2
        T_eq = beta * np.sqrt(Rs/(2*a)) * T_star
        taus = k_th * self.pressures / g

        def incoming_stream_contribution(gamma):
            return 3.0/4 * T_eq**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*np.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * scipy.special.expn(2, gamma*taus))

        e1 = incoming_stream_contribution(gamma)
        T4 = 3.0/4 * T_int**4 * (2.0/3 + taus) + (1 - alpha) * e1

        if gamma2 is not None:
            e2 = incoming_stream_contribution(gamma2)
            T4 += alpha * e2
        self.temperatures = T4 ** 0.25
