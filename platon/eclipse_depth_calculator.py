import numpy as np
import matplotlib.pyplot as plt

from .constants import h, c, k_B

class EclipseDepthCalculator:
    def compute_depths(self, lambda_grid, absorption_coeff_atm, radii, T_profile, P_profile, T_star, transit_depths, num_mu=100):
        #cutoff = len(T_profile) * 0.7
        #for i in range(len(T_profile)):
        #    if i < cutoff: continue
        #    T_profile[i] += (i - cutoff) * 10
        #plt.semilogy(T_profile, P_profile)
        #plt.show()

        intermediate_coeff = 0.5 * (absorption_coeff_atm[:, 0:-1] + absorption_coeff_atm[:, 1:])
        intermediate_T = 0.5 * (T_profile[0:-1] + T_profile[1:])
        dr = -np.diff(radii)
        d_taus = intermediate_coeff * dr
        taus = np.cumsum(d_taus, axis=1)

        mu_grid = np.linspace(1e-3, 1, num_mu)
        d_mu = 1.0/(num_mu - 1)
    
        fluxes = []        
        for i, w in enumerate(lambda_grid):
            all_B = 2*h*c**2/w**5/np.exp(h*c/w/k_B/intermediate_T - 1)
            exp_factor = np.exp(-taus[i].reshape((-1, 1))/mu_grid)
            integrands = exp_factor * all_B.reshape((-1, 1))
            integral = np.sum(integrands * d_taus[i].reshape((-1, 1)) * d_mu)
            fluxes.append(2*np.pi*integral)

        fluxes = np.array(fluxes)
        #blackbody_planet = np.pi * 2*h*c**2/lambda_grid**5/np.exp(h*c/lambda_grid/k_B/1200 - 1)
        blackbody_star = np.pi * 2*h*c**2/lambda_grid**5/np.exp(h*c/lambda_grid/k_B/T_star - 1)
        eclipse_depths = fluxes / blackbody_star * transit_depths
        return eclipse_depths


