import matplotlib.pyplot as plt
import numpy as np

from platon.constants import h, c, k_B, R_jup, M_jup, R_sun, PC
from platon.flux_calculator import FluxCalculator
from platon.TP_profile import Profile

Mp = 4 * M_jup
Rp = 1 * R_jup
log_k_th = -0.7
T_int = 300
dist = 22.4 * PC
P_quench = 1e4

calc = FluxCalculator(include_condensation=True)

#Uncomment below to get binned eclipse depths
#edges = np.linspace(1.1e-6, 1.7e-6, 30)
#bins = np.array([edges[0:-1], edges[1:]]).T
#calc.change_wavelength_bins(bins)

#Create T/P profile
p = Profile()

#Guillot et al parameterization with only internal heat
p.set_from_radiative_solution(Mp, Rp, log_k_th, T_int)

#Can manually modify T/P profile
p.temperatures[p.temperatures > 3000] = 3000

wavelengths, depths, info_dict = calc.compute_fluxes(
    p, Mp, Rp, dist, P_quench=P_quench, full_output=True)

plt.semilogy(p.temperatures, p.pressures)
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (Pa)")
plt.gca().invert_yaxis()
plt.figure()

plt.loglog(1e6*wavelengths, 1e6*depths)
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("Eclipse depth (ppm)")
plt.show()
