from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from platon.transit_depth_calculator import TransitDepthCalculator

# All quantities in SI
Rs = 6.57e8     #Radius of star
g = 21.7        #Planet's surface gravity at 1 bar (by default)
Rp = 1.16e7     #Radius of planet
T = 2150        #Temperature of isothermal part of the atmosphere

#create a TransitDepthCalculator object and compute wavelength dependent transit depths
depth_calculator = TransitDepthCalculator(Rs, g)
wavelengths, transit_depths = depth_calculator.compute_depths(Rp, T)

print("#Wavelength(m)       Depth")
for i in range(len(wavelengths)):
    print(wavelengths[i], transit_depths[i])

plt.plot(wavelengths, transit_depths)
plt.show()
