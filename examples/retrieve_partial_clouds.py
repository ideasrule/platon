"""Retrieval with a partially cloudy terminator and a non-isothermal T/P
profile, using pymultinest.

This example generates synthetic JWST/PRISM-like data (0.8-5.3 um at R=100)
for a hot Jupiter whose terminator is 70% covered by clouds and follows a
parametric (Madhusudhan & Seager 2009) T/P profile, then retrieves the
planet radius, composition (logZ and C/O), cloud properties (cloudtop
pressure and cloud fraction), and all six T/P profile parameters.

The transit T/P profile is controlled by transit_profile_type, and its
parameters carry a "_transit" suffix (T0_transit, log_P1_transit, ...);
any parameter without a "_transit" version falls back to the unsuffixed
(dayside) value.  cloud_fraction blends a cloudy and a clear terminator:
depth = cloud_fraction * cloudy + (1 - cloud_fraction) * clear, where the
clear spectrum has no cloud deck, haze, or Mie scattering, but keeps
Rayleigh scattering.
"""
import numpy as np
import pickle

from platon.combined_retriever import CombinedRetriever
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile
from platon.plotter import Plotter
from platon.constants import R_sun, R_jup, M_jup

# --- Synthetic PRISM-like observation: R = 100 from 0.8 to 5.3 um ---
R = 100
edges = [0.8e-6]
while edges[-1] < 5.3e-6:
    edges.append(edges[-1] * (1 + 1.0 / R))
edges = np.array(edges)
bins = np.column_stack([edges[:-1], edges[1:]])

# True planet: HD 209458b-like, 70% cloudy, hotter at depth
Rs = 1.19 * R_sun
Mp = 0.73 * M_jup
Rp = 1.4 * R_jup
T_star = 6091
T0, log_P1, alpha1, alpha2, log_P3, T3 = 850, 2.4, 2, 2, 6, 1400

true_profile = Profile()
true_profile.set_parametric(T0, 10.0**log_P1, alpha1, alpha2,
                            10.0**log_P3, T3)

calc = TransitDepthCalculator()
calc.change_wavelength_bins(bins)
_, true_depths, _ = calc.compute_depths(
    true_profile, Rs, Mp, Rp, logZ=0, CO_ratio=0.53,
    cloudtop_pressure=1e4, cloud_fraction=0.7, T_star=T_star)

rng = np.random.default_rng(42)
errors = np.full(len(bins), 50e-6)
depths = true_depths + rng.normal(0, errors)

# --- Set up the retrieval ---
retriever = CombinedRetriever()

# Note: no T is given; the terminator temperature structure comes entirely
# from the parametric profile parameters
fit_info = retriever.get_default_fit_info(
    Rs=Rs, Mp=Mp, Rp=Rp,
    logZ=0, CO_ratio=0.53, log_cloudtop_P=4, cloud_fraction=0.7,
    log_scatt_factor=0, scatt_slope=4, error_multiple=1, T_star=T_star,
    transit_profile_type="parametric",
    T0_transit=T0, log_P1_transit=log_P1, alpha1_transit=alpha1,
    alpha2_transit=alpha2, log_P3_transit=log_P3, T3_transit=T3)

fit_info.add_gaussian_fit_param('Rs', 0.02 * R_sun)
fit_info.add_gaussian_fit_param('Mp', 0.04 * M_jup)

fit_info.add_uniform_fit_param('Rp', 0.9 * Rp, 1.1 * Rp)
fit_info.add_uniform_fit_param('logZ', -1, 3)
fit_info.add_uniform_fit_param('CO_ratio', 0.2, 1.5)
fit_info.add_uniform_fit_param('log_cloudtop_P', -0.99, 5)
fit_info.add_uniform_fit_param('cloud_fraction', 0, 1)
fit_info.add_uniform_fit_param('error_multiple', 0.5, 5)

# The terminator T/P profile
fit_info.add_uniform_fit_param('T0_transit', 400, 1600)
fit_info.add_uniform_fit_param('log_P1_transit', 1, 4)
fit_info.add_uniform_fit_param('alpha1_transit', 0.1, 4)
fit_info.add_uniform_fit_param('alpha2_transit', 0.1, 4)
fit_info.add_uniform_fit_param('log_P3_transit', 4, 7)
fit_info.add_uniform_fit_param('T3_transit', 800, 2500)

result = retriever.run_multinest(
    bins, depths, errors,
    None, None, None,
    fit_info, rad_method="xsec", nlive=1000,
    multinest_kwargs=dict(outputfiles_basename="multinest_partial_clouds_"))

with open("partial_clouds_retrieval_result.pkl", "wb") as f:
    pickle.dump(result, f)

# --- Plot the best fit and the posterior ---
plotter = Plotter()
plotter.plot_retrieval_transit_spectrum(result, prefix="partial_clouds")
plotter.plot_retrieval_corner(result, filename="partial_clouds_corner.png")
