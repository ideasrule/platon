import numpy as np
import matplotlib.pyplot as plt
import pickle

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_earth, M_earth
from platon.plotter import Plotter

waves, hw, depths, errors = np.loadtxt("k218b_renyu_combined.txt", unpack=True)
bins = 1e-6 * np.array([waves - hw, waves + hw]).T
retriever = CombinedRetriever()

fit_info = retriever.get_default_fit_info(
    Rs=0.4445 * R_sun, Mp=8.63 * M_earth, Rp=2.6 * R_earth, T=250,
    logZ=None, CO_ratio=None,
    fit_vmr=True,
    log_cloudtop_P=4,
    log_scatt_factor=0, scatt_slope=4, error_excess=0,
    T_star=3457, T_spot=3000, spot_cov_frac=0.05,
    transit_offsets={"offset_niriss": (0, 405),
                     "offset_g235h_nrs2": (535, 735),
                     "offset_g395h_nrs1": (735, 945),
                     "offset_g395h_nrs2": (945, len(depths))}
)

#fit_info.add_uniform_fit_param('Mp', 4*M_earth, 15*M_earth)
fit_info.add_uniform_fit_param('Rp', 2.4 * R_earth, 2.8 * R_earth)
fit_info.add_uniform_fit_param('T', 200, 500)
fit_info.add_uniform_fit_param("log_cloudtop_P", 0, 8)
fit_info.add_uniform_fit_param("offset_niriss", -200e-6, 200e-6)
fit_info.add_uniform_fit_param("offset_g235h_nrs2", -200e-6, 200e-6)
fit_info.add_uniform_fit_param("offset_g395h_nrs1", -200e-6, 200e-6)
fit_info.add_uniform_fit_param("offset_g395h_nrs2", -200e-6, 200e-6)
fit_info.add_uniform_fit_param("T_spot", 2000, 3457)
fit_info.add_uniform_fit_param("spot_cov_frac", 0, 0.2)

fit_info.add_gases_vmr(["CH4", "CO2", "H2O", "NH3", "HCN", "CO", "H2-He"], 1e-12, 10**-0.3)

#Use Nested Sampling to do the fitting
result = retriever.run_multinest(bins, depths, errors,
                                 None, None, None,
                                 fit_info,
                                 #sample="rwalk",
                                 rad_method="xsec",
                                 nlive=1000
                                 )
with open("retrieval_result_k2_18b.pkl", "wb") as f:
    pickle.dump(result, f)

Plotter.plot_retrieval_transit_spectrum(result, prefix="best_fit")
Plotter.plot_retrieval_corner(result, filename="corner.png")
Plotter.plot_contrib_func(result.best_fit_transit_dict, prefix="contrib")
