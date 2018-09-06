import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne

from .constants import h, c, k_B

class EclipseDepthCalculator:
    def compute_depths(self, lambda_grid, absorption_coeff_atm, radii, T_profile, P_profile, T_star, transit_depth, num_mu=100):
        start = time.time()
        intermediate_coeff = 0.5 * (absorption_coeff_atm[:, 0:-1] + absorption_coeff_atm[:, 1:])
        intermediate_T = 0.5 * (T_profile[0:-1] + T_profile[1:])
        dr = -np.diff(radii)
        d_taus = intermediate_coeff * dr
        taus = np.cumsum(d_taus, axis=1)

        mu_grid = np.linspace(1e-3, 1, num_mu)
        d_mu = 1.0/(num_mu - 1)
        print time.time() - start
        reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
        planck_function = ne.evaluate("2*h*c**2/reshaped_lambda_grid**5/exp(h*c/reshaped_lambda_grid/k_B/intermediate_T - 1)")
        print time.time() - start
        reshaped_taus = taus[:,:,np.newaxis]
        reshaped_planck = planck_function[:,:,np.newaxis]
        reshaped_d_taus = d_taus[:,:,np.newaxis]
        integrands = ne.evaluate("exp(-reshaped_taus/mu_grid) * reshaped_planck * reshaped_d_taus")
        print time.time() - start
        fluxes = 2 * np.pi * np.sum(integrands, axis=(1, 2)) * d_mu
        print time.time() - start
        blackbody_star = np.pi * 2*h*c**2/lambda_grid**5/np.exp(h*c/lambda_grid/k_B/T_star - 1)
        eclipse_depths = fluxes / blackbody_star * transit_depth
        return eclipse_depths
