import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import corner
import pickle
import sys

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup, AU
from platon.plotter import Plotter

spec_filename = "hd189_combined_eclipse.txt"
output_dirname = "hd189_eclipse/"

start_waves, end_waves, depths, low_errs, high_errs, _ = np.loadtxt(spec_filename, unpack=True)
errs = (low_errs + high_errs) / 2
bins = 1e-6 * np.array([start_waves, end_waves]).T

retriever = CombinedRetriever()

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
    Rs=0.75 * R_sun, Mp=1.123 * M_jup, Rp=1.117 * R_jup,
    logZ=1, CO_ratio=0.7, log_cloudtop_P=np.inf,
    log_scatt_factor=0, scatt_slope=4, error_excess=0, T_star=5052,
    T_irr = 1692,
    log_k_th = -2.52, log_gamma=-0.8, log_gamma2=-0.8, alpha=0.4, beta=1, T_int=100,
    profile_type="line2013",
    eclipse_offsets={"offset_f322": (0, 31)}
)

fit_info.add_uniform_fit_param("logZ", -1, 3)
fit_info.add_uniform_fit_param("CO_ratio", 0.2, 2)
fit_info.add_uniform_fit_param("log_CH4_mult", -6, 3)

fit_info.add_uniform_fit_param("log_k_th", -5, 0)
fit_info.add_uniform_fit_param("log_gamma", -4, 2)
fit_info.add_uniform_fit_param("log_gamma2", -4, 2)
fit_info.add_uniform_fit_param("alpha", 0, 0.5)
fit_info.add_uniform_fit_param("beta", 0.5, 2)
#fit_info.add_uniform_fit_param("T_int", 50, 500)

fit_info.add_uniform_fit_param("offset_f322", -100e-6, 100e-6)
fit_info.add_uniform_fit_param("error_excess", 0, 100e-6)

result = retriever.run_multinest(None, None, None,
                                 bins, depths, errs,
                                 fit_info, nlive=1000,
                                 include_condensation=True,
                                 #zero_opacities=["H2S"]
                                 )

if not os.path.exists(output_dirname):
    os.makedirs(output_dirname)

with open(output_dirname + "retrieval_result.pkl", "wb") as f:
    pickle.dump(result, f)


plotter = Plotter()
#Plot the spectrum and save it to best_fit.png
plotter.plot_retrieval_eclipse_spectrum(result, prefix=output_dirname + 'best_fit')

#Plot the 2D posteriors with "corner" package and save it to multinest_corner.png
plotter.plot_retrieval_corner(result, filename=output_dirname + "corner.png")

#Plot the contribution function
plotter.plot_contrib_func(result.best_fit_eclipse_dict, prefix=output_dirname + 'best_fit')

#Plot the retrieved TP profiles
plotter.plot_retrieval_TP_profiles(result, plot_samples=True, plot_1sigma_bounds=False, prefix=output_dirname + 'corner')
