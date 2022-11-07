import matplotlib.pyplot as plt
import numpy as np
np.random.seed(20)

from platon.constants import h, c, k_B, R_jup, M_jup, R_sun
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.TP_profile import Profile

p = Profile()
p.set_parametric(1200, 500, 0.5, 0.6, 1e6, 1900)
#p.set_isothermal(1500)
calc = EclipseDepthCalculator(method="xsec") #"ktables" for correlated k

#Uncomment below to get binned eclipse depths
#edges = np.linspace(1.1e-6, 1.7e-6, 30)
#bins = np.array([edges[0:-1], edges[1:]]).T
#calc.change_wavelength_bins(bins)

for i in range(10):
    print(i)
    M_p = M_jup * np.random.uniform(0.9, 1.1)
    p.temperatures *= np.random.uniform(0.9, 1.1)
    wavelengths, depths, info_dict = calc.compute_depths(p, R_sun, M_p, R_jup, 5700, full_output=True)
    
'''plt.semilogy(p.temperatures, p.pressures)
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (Pa)")
plt.figure()

plt.loglog(1e6*wavelengths, 1e6*depths)
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("Eclipse depth (ppm)")
plt.show()
'''
