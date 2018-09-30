from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.special
import numpy as np

from pkg_resources import resource_filename

from .constants import h, c

class Profile:
    def __init__(self):
        self.pressures = np.load(
            resource_filename(__name__, "data/pressures.npy"))

    def set_from_arrays(self, P_profile, T_profile):        
        interpolator = interp1d(np.log10(P_profile), T_profile)
        self.temperatures = interpolator(self.pressures)
        
    def set_isothermal(self, T):
        self.temperatures = np.ones(len(self.pressures)) * T
        
    def set_parametric(self, T0, P1, alpha1, alpha2, P3, T3):
        # Parametric model from https://arxiv.org/pdf/0910.1347.pdf
        P0 = np.min(self.pressures)
        #T2 = T3 - (alpha2**2 * T0 + alpha2**2/alpha1**2*np.log(P1/P0)**2 - alpha2**2 * T3 - np.log(P1/P3)**2)/ (4 * np.log(P1/P3)**2 * alpha2**2)
        #P2 = P3/np.exp(alpha2 * (T3 - T2)**0.5)

        ln_P2 = alpha2**2*(T0+np.log(P1/P0)**2/alpha1**2 - T3) - np.log(P1)**2 + np.log(P3)**2
        ln_P2 /= 2 * np.log(P3/P1)
        P2 = np.exp(ln_P2)
        T2 = T3 - np.log(P3/P2)**2/alpha2**2


        print T0 + np.log(P1/P0)**2/alpha1**2, T2 + np.log(P1/P2)**2/alpha2**2
        print T2 + np.log(P3/P2)**2/alpha2**2, T3
        
        print P2, T2
        self.temperatures = np.zeros(len(self.pressures))
        for i, P in enumerate(self.pressures):
            if P < P1:
                self.temperatures[i] = T0 + np.log(P/P0)**2 / alpha1**2
            elif P < P3:
                self.temperatures[i] = T2 + np.log(P/P2)**2 / alpha2**2
            else:
                self.temperatures[i] = T3

    def set_from_opacity(self, Tirr, info_dict, visible_cutoff=0.8e-6, Tint=100):
        wavelengths = info_dict["unbinned_wavelengths"]
        stellar_spectrum = info_dict["stellar_spectrum"] * h * c / wavelengths
        planet_spectrum = info_dict["planet_spectrum"] * h * c / wavelengths
        absorption_coeffs = info_dict["absorption_coeff_atm"]
        radii = info_dict["radii"]
                
        # Equation 49 here: https://arxiv.org/pdf/1006.4702.pdf
        visible = wavelengths < visible_cutoff
        thermal = wavelengths >= visible_cutoff
        intermediate_coeffs = 0.5 * (absorption_coeffs[:, 0:-1] + absorption_coeffs[:, 1:])
        k_v = np.average(intermediate_coeffs[visible], axis=0, weights=stellar_spectrum[visible])
        k_th = np.average(intermediate_coeffs[thermal], axis=0, weights=planet_spectrum[thermal])
        gamma = k_v / k_th
        dr = -np.diff(radii)
        d_taus = k_th * dr
        taus = np.cumsum(d_taus)
        print taus
        e2 = scipy.special.expn(2, gamma*taus) #np.exp(-gamma*taus) - gamma * taus * scipy.special.gammainc(0, gamma*taus)
        T4 = 3.0/4 * Tint**4 * (2.0/3 + taus) + 3.0/4 * Tirr**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*np.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * e2)
        T = T4 ** 0.25
        self.temperatures = np.append(T[0], T)

'''profile = Profile(np.logspace(-4, 9, 500))
profile.set_parametric(1200, 500, 0.5, 0.6, 1e6, 1900)
plt.semilogy(profile.temperatures, profile.pressures)
plt.gca().invert_yaxis()
plt.show()'''
