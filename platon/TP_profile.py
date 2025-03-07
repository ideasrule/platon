import matplotlib.pyplot as plt
from . import _cupy_numpy as xp
expn=xp.scipy.special.expn

from pkg_resources import resource_filename

from .constants import h, c, k_B, AMU, G
from .params import NUM_LAYERS, MIN_P, MAX_P

class Profile:
    def __init__(self):
        self.pressures = xp.logspace(
                xp.log10(MIN_P),
                xp.log10(MAX_P),
                NUM_LAYERS)

    def get_temperatures(self):
        return xp.cpu(self.temperatures)

    def get_pressures(self):
        return xp.cpu(self.pressures)

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
        elif profile_type == "rates":
            self.set_from_rates(**params_dict)
        else:
            assert(False)

    def set_from_rates(self, Tmin, rn4, rn3, rn2, rn1, r0, r1, **ignored_kwargs):
        P_bar = self.pressures * 1e-5
        P_knots = xp.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        r_knots = xp.array([rn4, rn3, rn2, rn1, r0, r1])
        rates = xp.interp(xp.log10(P_bar), xp.log10(P_knots), r_knots, left=0, right=0)
        
        temperatures = []
        for i in range(len(P_bar)):
            if i == 0:
                temperatures.append(xp.array(Tmin))
            else:
                temperatures.append(temperatures[-1] + rates[i] * (xp.log10(P_bar[i] / P_bar[i-1])))

        self.temperatures = xp.array(temperatures)
        #plt.semilogy(self.temperatures.get(), self.pressures.get())
        #plt.gca().invert_yaxis()
        #plt.show()
                                        
        
    def set_from_arrays(self, P_profile, T_profile):
        P_profile = xp.array(P_profile)
        T_profile = xp.array(T_profile)
        self.temperatures = xp.interp(xp.log10(self.pressures), xp.log10(P_profile), T_profile)

    def set_isothermal(self, T_day):
        self.temperatures = xp.ones(len(self.pressures)) * T_day

    def set_parametric(self, T0, P1, alpha1, alpha2, P3, T3):
        '''Parametric model from https://arxiv.org/pdf/0910.1347.pdf'''
        P0 = xp.amin(self.pressures)

        ln_P2 = alpha2**2*(T0+xp.log(P1/P0)**2/alpha1**2 - T3) - xp.log(P1)**2 + xp.log(P3)**2
        ln_P2 /= 2 * xp.log(P3/P1)
        P2 = xp.exp(ln_P2)
        T2 = T3 - xp.log(P3/P2)**2/alpha2**2

        self.temperatures = xp.zeros(len(self.pressures))
        for i, P in enumerate(self.pressures):
            if P < P1:
                self.temperatures[i] = T0 + xp.log(P/P0)**2 / alpha1**2
            elif P < P3:
                self.temperatures[i] = T2 + xp.log(P/P2)**2 / alpha2**2
            else:
                self.temperatures[i] = T3
        return P2, T2

    def set_from_opacity(self, T_irr, info_dict, visible_cutoff=0.8e-6, T_int=100):
        wavelengths = xp.array(info_dict["unbinned_wavelengths"])
        d_lambda = xp.diff(wavelengths)
        d_lambda = xp.append(d_lambda[0], d_lambda)

        # Convert stellar spectrum from photons/time to energy/time
        stellar_spectrum = xp.array(info_dict["stellar_spectrum"]) * h * c / wavelengths

        # Convert planetary spectrum from energy/time/wavelength to energy/time
        planet_spectrum = xp.array(info_dict["planet_spectrum"]) * d_lambda
        absorption_coeffs = xp.array(info_dict["absorption_coeff_atm"])
        radii = xp.array(info_dict["radii"])

        # Equation 49 here: https://arxiv.org/pdf/1006.4702.pdf
        visible = wavelengths < visible_cutoff
        thermal = wavelengths >= visible_cutoff
        n = xp.array(info_dict["P_profile"]/k_B/info_dict["T_profile"])
        intermediate_n = (n[0:-1] + n[1:])/2.0
        sigmas = absorption_coeffs / n[:, xp.newaxis]
        sigma_v = xp.median(xp.average(sigmas[:, visible], axis=1, weights=stellar_spectrum[visible]))
        sigma_th = xp.median(xp.average(sigmas[:, thermal], axis=1, weights=planet_spectrum[thermal]))

        gamma = sigma_v / sigma_th

        dr = -xp.diff(radii)
        d_taus = sigma_th * intermediate_n * dr
        taus = xp.cumsum(d_taus)

        e2 = expn(2, gamma*taus)
        T4 = 3.0/4 * T_int**4 * (2.0/3 + taus) + 3.0/4 * T_irr**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*xp.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * e2)
        T = T4 ** 0.25
        self.temperatures = xp.append(T[0], T)

    def set_from_radiative_solution(self, T_irr, Mp, Rp, beta, log_k_th, log_gamma, log_gamma2=None, alpha=0, T_int=100, **ignored_kwargs):
        '''From Line et al. 2013: http://adsabs.harvard.edu/abs/2013ApJ...775..137L, Equation 13 - 16'''

        g = G * Mp / Rp**2
        k_th = 10.0**log_k_th
        gamma = 10.0**log_gamma
        gamma2 = 10.0**log_gamma2
        
        T_eq = beta * T_irr / xp.sqrt(2) 
        taus = k_th * self.pressures / g

        def incoming_stream_contribution(gamma):
            return 3.0/4 * T_eq**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*xp.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * expn(2, gamma*taus))

        e1 = incoming_stream_contribution(gamma)
        T4 = 3.0/4 * T_int**4 * (2.0/3 + taus) + (1 - alpha) * e1

        if gamma2 is not None:
            e2 = incoming_stream_contribution(gamma2)
            T4 += alpha * e2
        self.temperatures = T4 ** 0.25
