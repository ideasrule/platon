import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from platon.fit_info import FitInfo
from platon.retriever import Retriever
from platon.constants import R_sun, R_jup, M_jup, AU, PC, h, c
from scipy.ndimage import uniform_filter

wavelengths, fluxes, errors = np.loadtxt(sys.argv[1], unpack=True, skiprows=1, delimiter=",")
#plt.errorbar(wavelengths, fluxes, yerr=errors)
#plt.show()

wavelengths *= 1e-6
fluxes *= 1e6
errors *= 1e6

#Restrict to first segment
cond = np.logical_and(wavelengths > 1e-6, wavelengths < 1.35e-6)
wavelengths = wavelengths[cond]
fluxes = fluxes[cond]
errors = errors[cond]

#Downsample by 100x
factor = 100
wavelengths = wavelengths[::factor]
fluxes = uniform_filter(fluxes, factor)[::factor]
errors = uniform_filter(errors, factor)[::factor] * 2
width = np.median(np.diff(wavelengths))
bins = np.array([wavelengths - width/2, wavelengths + width/2]).T

#Fluxes to photon fluxes
#factor = width / (h * c / wavelengths)
#fluxes *= factor
#errors *= factor

plt.errorbar(wavelengths, fluxes, yerr=errors, fmt='.')
#plt.figure()
#plt.plot(np.diff(wavelengths))
#plt.show()

#create a Retriever object
retriever = Retriever()

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
    Mp=9 * M_jup, Rp=2.43 * R_jup,
    logZ=0, CO_ratio=0.53, log_cloudtop_P=np.inf,
    log_scatt_factor=0, scatt_slope=4, error_multiple=1,
    dist=144.159*PC,
    a = None,
    T_int=100,
    log_k_th = -2.52, log_gamma=-0.8, log_gamma2=-0.8, alpha=0.5, beta=None,
    profile_type="radiative_solution" #"isothermal" for isothermal fitting
    )

#Add fitting parameters - this specifies which parameters you want to fit
#e.g. since we have not included cloudtop_P, it will be fixed at the value specified in the constructor

#Physical parameters
fit_info.add_uniform_fit_param("Mp", 5*M_jup, 30*M_jup)
fit_info.add_uniform_fit_param("Rp", 1*R_jup, 4*R_jup)

#Chemistry
#fit_info.add_uniform_fit_param("logZ", -1, 3)
#fit_info.add_uniform_fit_param("CO_ratio", 0.2, 2)

#T/P profile parameters
#fit_info.add_uniform_fit_param("T", 100, 3000)
fit_info.add_uniform_fit_param("T_int", 100, 2500)
fit_info.add_uniform_fit_param("log_k_th", -5, 0)
#fit_info.add_uniform_fit_param("log_gamma", -4, 1)
#fit_info.add_uniform_fit_param("log_gamma2", -4, 1)
#fit_info.add_uniform_fit_param("alpha", 0, 0.5)


#Use Nested Sampling to do the fitting
result = retriever.run_multinest(bins, fluxes, errors,
                                 fit_info, plot_best=True, nlive=200,
                                 sample="rwalk",
                                 rad_method="xsec") #"ktables" instead of "xsec" for correlated k

with open("example_retrieval_result.pkl", "wb") as f:
    pickle.dump(result, f)

#Plot the spectrum and save it to best_fit.png
result.plot_spectrum("best_fit")

#Plot the 2D posteriors with "corner" package and save it to multinest_corner.png
result.plot_corner("multinest_corner.png")
