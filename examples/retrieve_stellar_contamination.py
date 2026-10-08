import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.combined_retriever import CombinedRetriever
from platon.TP_profile import Profile
from platon.constants import R_sun, M_jup, R_jup
from platon.plotter import Plotter

# An M dwarf with unocculted spots (cooler than the photosphere) and faculae
# (hotter).  Both are "heterogeneities": T_het/het_cov_frac is the first,
# T_het2/het2_cov_frac the second; nothing assumes which is cooler.
Rs, Mp, Rp, T = 0.4 * R_sun, 0.3 * M_jup, 0.8 * R_jup, 700
star = dict(T_star=3400, T_het=3000, het_cov_frac=0.05, T_het2=3700, het2_cov_frac=0.03)

# Synthetic NIRSpec PRISM-like spectrum, 100 ppm errors
edges = np.geomspace(0.6e-6, 5.2e-6, 120)
bins = np.column_stack([edges[:-1], edges[1:]])
calculator = TransitDepthCalculator(stellar_grid="newera", logg_star=4.9, feh_star=0)
calculator.change_wavelength_bins(bins)
_, depths, _ = calculator.compute_depths(Profile.isothermal(T), Rs, Mp, Rp, **star)
errors = np.full(len(depths), 100e-6)
depths += np.random.default_rng(0).normal(0, errors)

# Fit the planet and both heterogeneities.  Keep the two temperature priors
# on opposite sides of T_star, so the spots and faculae cannot swap labels.
retriever = CombinedRetriever()
fit_info = retriever.get_default_fit_info(
    Rs, Mp, 0.9 * Rp, T, logZ=0, CO_ratio=0.53,
    stellar_grid="newera", logg_star=4.9, feh_star=0, **star)
fit_info.add_uniform_fit_param("Rp", 0.7 * R_jup, 0.9 * R_jup)
fit_info.add_uniform_fit_param("T", 400, 1000)
fit_info.add_uniform_fit_param("T_het", 2300, 3400)       # spots
fit_info.add_uniform_fit_param("het_cov_frac", 0, 0.3)
fit_info.add_uniform_fit_param("T_het2", 3400, 4400)      # faculae
fit_info.add_uniform_fit_param("het2_cov_frac", 0, 0.3)

result = retriever.run_dynesty(bins, depths, errors, None, None, None, fit_info, nlive=100)
Plotter.plot_retrieval_transit_spectrum(result, prefix="stellar_contamination")
Plotter.plot_retrieval_corner(result, filename="stellar_contamination_corner.png")
