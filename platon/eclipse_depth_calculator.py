import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne
import gnumpy as gnp

from .constants import h, c, k_B

class EclipseDepthCalculator:
    def compute_depths(self, lambda_grid, absorption_coeff_atm, radii, T_profile, P_profile, T_star, transit_depth, min_mu=1e-3, max_mu=1, num_mu=100):
        start = time.time()
        intermediate_coeff = 0.5 * (absorption_coeff_atm[:, 0:-1] + absorption_coeff_atm[:, 1:])
        intermediate_T = 0.5 * (T_profile[0:-1] + T_profile[1:])
        dr = -np.diff(radii)
        d_taus = intermediate_coeff * dr
        taus = np.cumsum(d_taus, axis=1)

        mu_grid = np.linspace(min_mu, max_mu, num_mu)
        d_mu = (max_mu - min_mu)/(num_mu - 1)
        print time.time() - start
        reshaped_lambda_grid = gnp.garray(lambda_grid.reshape((-1, 1)))
        planck_function = 2*h*c**2/reshaped_lambda_grid**5/gnp.exp(h*c/reshaped_lambda_grid/k_B/gnp.garray(intermediate_T) - 1)
        
        print time.time() - start
        reshaped_taus = gnp.garray(taus[:,:,np.newaxis])
        reshaped_planck = planck_function[:,:,np.newaxis]
        reshaped_d_taus = gnp.garray(d_taus[:,:,np.newaxis])

        integrands = gnp.exp(-reshaped_taus/gnp.garray(mu_grid)) * reshaped_planck * reshaped_d_taus 
        print time.time() - start
        fluxes = integrands.sum(axis=1)
        fluxes = 2 * np.pi * d_mu * fluxes.sum(axis=1)
        fluxes = fluxes.asarray()
        print time.time() - start
        blackbody_star = np.pi * 2*h*c**2/lambda_grid**5/np.exp(h*c/lambda_grid/k_B/T_star - 1)
        eclipse_depths = fluxes / blackbody_star * transit_depth
        print eclipse_depths
        return eclipse_depths
