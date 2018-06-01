import numpy as np
import matplotlib.pyplot as plt
import corner

from pyexotransmit.fit_info import FitInfo
from pyexotransmit.transit_depth_calculator import TransitDepthCalculator
from pyexotransmit.retrieve import Retriever

Rs = 7e8
g = 9.8
Rp = 7.14e7
logZ = 0
CO = 0.53
log_cloudtop_P = 3
temperature = 1200


depth_calculator = TransitDepthCalculator(Rs, g)

wavelength_bins = []
stis_wavelengths = np.linspace(0.4e-6, 0.7e-6, 30)
for i in range(len(stis_wavelengths) - 1):
    wavelength_bins.append([stis_wavelengths[i], stis_wavelengths[i+1]])

wfc_wavelengths = np.linspace(1.1e-6, 1.7e-6, 30)
for i in range(len(wfc_wavelengths) - 1):
    wavelength_bins.append([wfc_wavelengths[i], wfc_wavelengths[i+1]])

wavelength_bins.append([3.2e-6, 4e-6])
wavelength_bins.append([4e-6, 5e-6])
depth_calculator.change_wavelength_bins(wavelength_bins)

wavelengths, transit_depths = depth_calculator.compute_depths(Rp, temperature, logZ=logZ, CO_ratio=CO, cloudtop_pressure=1e3)
#wavelengths, depths2 = depth_calculator.compute_depths(71414515.1348402, P_prof

retriever = Retriever()

fit_info = retriever.get_default_fit_info(Rs, g, 0.99*Rp, 0.9*temperature, logZ=2, CO_ratio=1, add_fit_params=True)

errors = np.random.normal(scale=50e-6, size=len(transit_depths))
transit_depths += errors

result = retriever.run_multinest(wavelength_bins, transit_depths, errors, fit_info)
np.save("samples.npy", result.samples)
np.save("weights.npy", result.weights)
np.save("logl.npy", result.logl)

print fit_info.fit_param_names
fig = corner.corner(result.samples, weights=result.weights, range=[0.99] * result.samples.shape[1], labels=fit_info.fit_param_names)
fig.savefig("multinest_corner.png")
