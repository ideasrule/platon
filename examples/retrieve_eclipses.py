import numpy as np
import matplotlib.pyplot as plt
import pickle

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup, AU

#Retrieval on HD 189733b eclipse data

def eclipse_wfc3():
    #https://arxiv.org/pdf/1409.4000.pdf
    wavelengths = 1e-6*np.array([1.1279, 1.1467, 1.1655, 1.1843, 1.2031, 1.2218, 1.2406, 1.2594, 1.2782, 1.2969, 1.3157, 1.3345, 1.3533, 1.3721, 1.3908, 1.4096, 1.4284, 1.4472, 1.466, 1.4848, 1.5035, 1.5223, 1.5411, 1.5599, 1.5786, 1.5974, 1.6162, 1.6350]) 
    wavelength_bins = [[w-0.0095e-6, w+0.0095e-6] for w in wavelengths]
    depths = 1e-6 * (96 + np.array([-96, -18, 28, -3, -7, -45, -32, 3, -16, 53, 31, 12, -21, -27, -40, 8, -2, -29, -64, 9, 132, 115, 47, 3, 39, -16, -64, 14]))
    errors = 1e-6 * np.array([47, 50, 45, 44, 43, 50, 42, 42, 42, 41, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 45, 45, 46, 46, 74])
    return np.array(wavelength_bins), depths, errors

def eclipse_spitzer():
    #From Charbonneau et al 2008, Agol et al 2010, Knutson et al 2012, Knutson et al 2009

    wave_bins = []
    depths = []
    errors = []

    wave_bins.append([3.2, 4.0])
    wave_bins.append([4.0, 5.0])
    wave_bins.append([5.1, 6.3])
    wave_bins.append([6.6, 9.0])
    wave_bins.append([13.5, 18.5])
    wave_bins.append([20.8, 26.1])
    depths = np.array([0.1481, 0.1827, 0.31, 0.344, 0.519, 0.536]) * 1e-2
    errors = np.array([0.0034, 0.0022, 0.034, 0.0036, 0.022, 0.027]) * 1e-2

    return 1e-6*np.array(wave_bins), depths, errors

wfc3_bins, wfc3_depths, wfc3_errors = eclipse_wfc3()
spitzer_bins, spitzer_depths, spitzer_errors = eclipse_spitzer()

eclipse_bins = np.concatenate([wfc3_bins, spitzer_bins])
eclipse_depths = np.concatenate([wfc3_depths, spitzer_depths])
eclipse_errors = np.concatenate([wfc3_errors, spitzer_errors])

R_guess = 1.113 * R_jup
T_star = 5052

#create a Retriever object
retriever = CombinedRetriever()

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
    Rs=0.75 * R_sun, Mp=1.123 * M_jup, Rp=R_guess,
    logZ=0, CO_ratio=0.53, log_cloudtop_P=np.inf,
    log_scatt_factor=0, scatt_slope=4, error_multiple=1, T_star=T_star,
    a = 0.03142 * AU,
    log_k_th = -2.52, log_gamma=-0.8, log_gamma2=-0.8, alpha=0.5, beta=1,
    profile_type="radiative_solution" #"isothermal" for isothermal fitting
    )

#Add fitting parameters - this specifies which parameters you want to fit
#e.g. since we have not included cloudtop_P, it will be fixed at the value specified in the constructor

#Chemistry
fit_info.add_uniform_fit_param("logZ", -1, 3)
fit_info.add_uniform_fit_param("CO_ratio", 0.2, 2)

#T/P profile parameters
fit_info.add_uniform_fit_param("log_k_th", -5, 0)
fit_info.add_uniform_fit_param("log_gamma", -4, 1)
fit_info.add_uniform_fit_param("log_gamma2", -4, 1)
fit_info.add_uniform_fit_param("alpha", 0, 0.5)
fit_info.add_uniform_fit_param("beta", 0, 2)

#Nuisance parameters
fit_info.add_gaussian_fit_param("wfc3_offset_eclipse", 39e-6)

#Use Nested Sampling to do the fitting
result = retriever.run_multinest(None, None, None,
                                 eclipse_bins, eclipse_depths, eclipse_errors,
                                 fit_info, nlive=200,
                                 sample="rwalk",
                                 rad_method="xsec") #"ktables" instead of "xsec" for correlated k

with open("example_retrieval_result.pkl", "wb") as f:
    pickle.dump(result, f)

#Plot the spectrum and save it to best_fit.png
result.plot_spectrum("best_fit")

#Plot the 2D posteriors with "corner" package and save it to multinest_corner.png
result.plot_corner("multinest_corner.png")
