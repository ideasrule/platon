from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import nestle
import corner

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup

def wfc3():
    #https://arxiv.org/pdf/1409.4000.pdf
    wavelengths = 1e-6*np.array([1.1279, 1.1467, 1.1655, 1.1843, 1.2031, 1.2218, 1.2406, 1.2594, 1.2782, 1.2969, 1.3157, 1.3345, 1.3533, 1.3721, 1.3908, 1.4096, 1.4284, 1.4472, 1.466, 1.4848, 1.5035, 1.5223, 1.5411, 1.5599, 1.5786, 1.5974, 1.6162, 1.6350]) 
    wavelength_bins = [[w-0.0095e-6, w+0.0095e-6] for w in wavelengths]
    depths = 1e-6 * (96 + np.array([-96, -18, 28, -3, -7, -45, -32, 3, -16, 53, 31, 12, -21, -27, -40, 8, -2, -29, -64, 9, 132, 115, 47, 3, 39, -16, -64, 14]))
    errors = 1e-6 * np.array([47, 50, 45, 44, 43, 50, 42, 42, 42, 41, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 45, 45, 46, 46, 74])
    errors = np.sqrt(errors**2 + (39e-6)**2)
    return np.array(wavelength_bins), depths, errors

def spitzer():
    #From Charbonneau et al 2008, Agol et al 2010, Knutson et al 2012

    wave_bins = []
    depths = []
    errors = []

    wave_bins.append([3.2, 4.0])
    wave_bins.append([4.0, 5.0])
    wave_bins.append([5.1, 6.3])
    wave_bins.append([6.6, 9.0])
    wave_bins.append([13.5, 18.5])
    wave_bins.append([20.8, 26.1])
    depths = np.array([0.1466, 0.1787, 0.31, 0.344, 0.391, 0.598]) * 1e-2
    errors = np.array([0.004, 0.0038, 0.034, 0.0036, 0.022, 0.038]) * 1e-2

    return 1e-6*np.array(wave_bins), depths, errors


wfc3_bins, wfc3_depths, wfc3_errors = wfc3()
spitzer_bins, spitzer_depths, spitzer_errors = spitzer()

bins = np.concatenate([wfc3_bins, spitzer_bins])
depths = np.concatenate([wfc3_depths, spitzer_depths])
errors = np.concatenate([wfc3_errors, spitzer_errors])

R_guess = 1.13 * R_jup

#create a Retriever object
retriever = CombinedRetriever()

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
    Rs=0.75 * R_sun, Mp=1.13 * M_jup, Rp=R_guess,
    logZ=0, CO_ratio=0.53, log_cloudtop_P=np.inf,
    log_scatt_factor=0, scatt_slope=4, error_multiple=1, T_star=5052,
    #T = 1295, #uncomment this for isothermal fitting
    T0=1295, log_P1=2.4, alpha1=2, alpha2=2, log_P3=6, T3=1295,
    profile_type="parametric" #"isothermal" for isothermal fitting
    )

#Add fitting parameters - this specifies which parameters you want to fit
#e.g. since we have not included cloudtop_P, it will be fixed at the value specified in the constructor

fit_info.add_gaussian_fit_param('Rs', 0.01*R_sun)
fit_info.add_gaussian_fit_param('Mp', 0.01*M_jup)

fit_info.add_uniform_fit_param('Rp', 0.9*R_guess, 1.1*R_guess)
fit_info.add_uniform_fit_param("logZ", -1, 3)
fit_info.add_uniform_fit_param("error_multiple", 0.5, 5)

fit_info.add_uniform_fit_param("T0", 1000, 3000)
fit_info.add_uniform_fit_param("log_P1", 1, 4)
fit_info.add_uniform_fit_param("alpha1", 0.1, 4)
fit_info.add_uniform_fit_param("alpha2", 0.1, 4)
fit_info.add_uniform_fit_param("log_P3", 5, 7)
fit_info.add_uniform_fit_param("T3", 1000, 3000)

# Uncomment below for isothermal fitting
#fit_info.add_uniform_fit_param("T", 1000, 3000)


#Use Nested Sampling to do the fitting
result = retriever.run_multinest(None, None, None,
                                 bins, depths, errors,
                                 fit_info, plot_best=True)
plt.savefig("best_fit.png")

np.save("samples.npy", result.samples)
np.save("weights.npy", result.weights)
np.save("logl.npy", result.logl)

fig = corner.corner(result.samples, weights=result.weights,
                    range=[0.99] * result.samples.shape[1],
                    labels=fit_info.fit_param_names)
fig.savefig("multinest_corner.png")

plt.show()
