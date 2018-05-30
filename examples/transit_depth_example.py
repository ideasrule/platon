import numpy as np
import matplotlib.pyplot as plt
import corner

from pyexotransmit.transit_depth_calculator import TransitDepthCalculator

# All quantities in SI
Rs = 6.57e8
g = 21.7
Rp = 1.16e7
T = 2200

depth_calculator = TransitDepthCalculator(Rs, g)
wavelengths, transit_depths = depth_calculator.compute_depths(Rp, T)

print "#Wavelength(m)       Depth"
for i in range(len(wavelengths)):
    print wavelengths[i], transit_depths[i]


