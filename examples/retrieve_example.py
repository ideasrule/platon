from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import nestle
import corner

from platon.fit_info import FitInfo
from platon.retriever import Retriever
from platon.constants import R_sun, R_jup, M_jup

def hd209458b_stis():
    #http://iopscience.iop.org/article/10.1086/510111/pdf
    star_radius = 1.125 * R_sun
    wave_bins = [[293,347], [348,402], [403,457], [458,512], [512,567], [532,629], [629,726], [727,824], [825,922], [922,1019]]
    wave_bins = 1e-9 * np.array(wave_bins)

    planet_radii = [1.3263, 1.3254, 1.32, 1.3179, 1.3177, 1.3246, 1.3176, 1.3158, 1.32, 1.3268]
    radii_errors = [0.0018, 0.0010, 0.0006, 0.0006, 0.0010, 0.0006, 0.0005, 0.0006, 0.0006, 0.0013]
    transit_depths = (np.array(planet_radii)*R_jup/star_radius)**2 + 60e-6
    transit_errors = np.array(radii_errors)/np.array(planet_radii) * 2 * transit_depths
    return wave_bins, transit_depths, transit_errors

def hd209458b_wfc3():
    #https://arxiv.org/pdf/1302.1141.pdf
    wavelengths = 1e-6*np.array([1.119, 1.138, 1.157, 1.175, 1.194, 1.213, 1.232, 1.251, 1.270, 1.288, 1.307, 1.326, 1.345, 1.364, 1.383, 1.401, 1.420, 1.439, 1.458, 1.477, 1.496, 1.515, 1.533, 1.552, 1.571, 1.590, 1.609, 1.628])
    wavelength_bins = [[w-0.0095e-6, w+0.0095e-6] for w in wavelengths]
    depths = 1e-6 * np.array([14512.7, 14546.5, 14566.3, 14523.1, 14528.7, 14549.9, 14571.8, 14538.6, 14522.2, 14538.4, 14535.9, 14604.5, 14685.0, 14779.0, 14752.1, 14788.8, 14705.2, 14701.7, 14677.7, 14695.1, 14722.3, 14641.4, 14676.8, 14666.2, 14642.5, 14594.1, 14530.1, 14642.1])
    errors = 1e-6 * np.array([50.6, 35.5, 35.2, 34.6, 34.1, 33.7, 33.5, 33.6, 33.8, 33.7, 33.4, 33.4, 33.5, 33.9, 34.4, 34.5, 34.7, 35.0, 35.4, 35.9, 36.4, 36.6, 37.1, 37.8, 38.6, 39.2, 39.9, 40.8])
    return np.array(wavelength_bins), depths, errors

def hd209458b_spitzer():
    #https://arxiv.org/pdf/1504.05942.pdf

    wave_bins = []
    depths = []
    errors = []

    wave_bins.append([3.2, 4.0])
    RpRs = np.average([0.12077, 0.1222, 0.11354, 0.11919], weights=1.0/np.array([0.00085, 0.00062, 0.00087, 0.00032]))
    depths.append(RpRs**2)
    errors.append(0.00032/RpRs * 2 * depths[-1])

    wave_bins.append([4.0, 5.0])
    RpRs = np.average([0.12199, 0.12099], weights=1.0/np.array([0.00094, 0.00029]))
    depths.append(RpRs**2)
    errors.append(0.00029/RpRs * 2 * depths[-1])

    wave_bins.append([5.1, 6.3])
    RpRs = np.average([0.12007, 0.11880], weights=1.0/np.array([0.00248, 0.00272]))
    depths.append(RpRs**2)
    errors.append(0.00248/RpRs * 2 * depths[-1])

    wave_bins.append([6.6, 9.0])
    RpRs = np.average([0.12007, 0.11991], weights=1.0/np.array([0.00114, 0.00073]))
    depths.append(RpRs**2)
    errors.append(0.00073/RpRs * 2 * depths[-1])

    return 1e-6*np.array(wave_bins), np.array(depths), np.array(errors)

stis_bins, stis_depths, stis_errors = hd209458b_stis()
wfc3_bins, wfc3_depths, wfc3_errors = hd209458b_wfc3()
spitzer_bins, spitzer_depths, spitzer_errors = hd209458b_spitzer()

bins = np.concatenate([stis_bins, wfc3_bins, spitzer_bins])
depths = np.concatenate([stis_depths, wfc3_depths, spitzer_depths])
errors = np.concatenate([stis_errors, wfc3_errors, spitzer_errors])

R_guess = 1.4 * R_jup
T_guess = 1200

#create a Retriever object
retriever = Retriever()

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
    Rs=1.19 * R_sun, Mp=0.73 * M_jup, Rp=R_guess, T=T_guess,
    logZ=0, CO_ratio=0.53, log_cloudtop_P=4,
    log_scatt_factor=0, scatt_slope=4, error_multiple=1)

#Add fitting parameters - this specifies which parameters you want to fit
#e.g. since we have not included cloudtop_P, it will be fixed at the value specified in the constructor

fit_info.add_gaussian_fit_param('Rs', 0.02*R_sun)
fit_info.add_gaussian_fit_param('Mp', 0.04*M_jup)

fit_info.add_uniform_fit_param('R', 0.9*R_guess, 1.1*R_guess)
fit_info.add_uniform_fit_param('T', 0.5*T_guess, 1.5*T_guess)
fit_info.add_uniform_fit_param("log_scatt_factor", 0, 1)
fit_info.add_uniform_fit_param("logZ", -1, 3)
fit_info.add_uniform_fit_param("log_cloudtop_P", -1, 5)
fit_info.add_uniform_fit_param("error_multiple", 0.5, 5)

#Use Nested Sampling to do the fitting
result = retriever.run_multinest(bins, depths, errors, fit_info, plot_best=True)

np.save("samples.npy", result.samples)
np.save("weights.npy", result.weights)
np.save("logl.npy", result.logl)

fig = corner.corner(result.samples, weights=result.weights,
                    range=[0.99] * result.samples.shape[1],
                    labels=fit_info.fit_param_names)
fig.savefig("multinest_corner.png")

