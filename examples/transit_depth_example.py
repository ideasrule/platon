from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import M_jup, R_sun, R_jup

# All quantities in SI
Rs = 1.16 * R_sun     #Radius of star
Mp = 0.73 * M_jup     #Mass of planet
Rp = 1.40 * R_jup      #Radius of planet
T = 1200              #Temperature of isothermal part of the atmosphere

#create a TransitDepthCalculator object and compute wavelength dependent transit depths
depth_calculator = TransitDepthCalculator()
wavelengths, transit_depths = depth_calculator.compute_depths(
    Rs, Mp, Rp, T, CO_ratio=0.2, cloudtop_pressure=1e4)


# Uncomment the code below to print

#print("#Wavelength(m)       Depth")
#for i in range(len(wavelengths)):
#    print(wavelengths[i], transit_depths[i])

# Uncomment the code below to plot

#plt.plot(1e6*wavelengths, transit_depths)
#plt.xlabel("Wavelength (um)")
#plt.ylabel("Transit depth")
#plt.show()
