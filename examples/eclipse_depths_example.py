import matplotlib.pyplot as plt
import numpy as np

from platon.constants import h, c, k_B, R_jup, M_jup, R_sun
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.TP_profile import Profile

edges = np.linspace(1.1e-6, 1.7e-6, 30)
bins = np.array([edges[0:-1], edges[1:]]).T

p = Profile()
p.set_parametric(1200, 500, 0.5, 0.6, 1e6, 1900)
#p.set_isothermal(1500)
calc = EclipseDepthCalculator()
#calc.change_wavelength_bins(bins)

wavelengths, depths, info_dict = calc.compute_depths(p, R_sun, M_jup, R_jup, 5700, full_output=True)
plt.semilogy(p.temperatures, p.pressures)
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (Pa)")
plt.figure()

plt.loglog(1e6*wavelengths, info_dict["planet_spectrum"])
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("Eclipse depth (ppm)")
plt.show()
