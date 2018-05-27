import numpy as np
import matplotlib.pyplot as plt
import corner

from fit_info import FitInfo
import eos_reader
from abundance_getter import AbundanceGetter
from transit_depth_calculator import TransitDepthCalculator
from retrieve import Retriever

Rs = 7e8
g = 9.8
Rp = 7.14e7
logZ = 0
CO = 0.53
log_cloudtop_P = 3
temperature = 1200


P_profile = np.logspace(np.log10(0.1), np.log10(2e5), 400)
T_profile = np.ones(400) * temperature

abund_getter = AbundanceGetter(include_condensates=True)
abundances = abund_getter.get(np.exp(logZ), CO)

depth_calculator = TransitDepthCalculator(Rs, g)
wfc_wavelengths = np.linspace(1.1e-6, 1.7e-6, 30)
wavelength_bins = []
for i in range(len(wfc_wavelengths) - 1):
    wavelength_bins.append([wfc_wavelengths[i], wfc_wavelengths[i+1]])

wavelength_bins.append([3.2e-6, 4e-6])
wavelength_bins.append([4e-6, 5e-6])
depth_calculator.change_wavelength_bins(wavelength_bins)

wavelengths, transit_depths = depth_calculator.compute_depths(Rp, P_profile, T_profile, abundances, cloudtop_pressure=1e3)
#wavelengths, depths2 = depth_calculator.compute_depths(71414515.1348402, P_prof
print transit_depths

retriever = Retriever(include_condensates=True)

fit_info = FitInfo({'R': 0.99*Rp, 'T': 0.9*temperature, 'logZ': np.log10(2), 'CO_ratio': 1, 'log_scatt_factor': np.log10(1), 'log_cloudtop_P': log_cloudtop_P+1, 'star_radius': Rs, 'g': g, 'error_multiple': 1})

fit_info.add_fit_param('R', 0.9*Rp, 1.1*Rp, 0, np.inf)
fit_info.add_fit_param('T', 0.5*temperature, 1.5*temperature, 0, np.inf)
fit_info.add_fit_param('logZ', -1, 3, -1, 3)
fit_info.add_fit_param('CO_ratio', 0.2, 1.5, 0.2, 2.0)
fit_info.add_fit_param('log_cloudtop_P', -1, 4, -np.inf, np.inf)
fit_info.add_fit_param('log_scatt_factor', 0, 1, 0, 3)
#fit_info.add_fit_param('error_multiple', 0.1, 10, 0, np.inf)

errors = np.random.normal(scale=1000e-6, size=len(transit_depths))
transit_depths += errors

result = retriever.run_multinest(wavelength_bins, transit_depths, errors, fit_info)
np.save("samples.npy", result.samples)
np.save("weights.npy", result.weights)
np.save("logl.npy", result.logl)
fig = corner.corner(result.samples, weights=result.weights, range=[0.99] * result.samples.shape[1], labels=fit_info.fit_param_names, truths=[Rp, temperature, logZ, CO, log_cloudtop_P, 0, 1])
fig.savefig("multinest_corner.png")
