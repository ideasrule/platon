import matplotlib.pyplot as plt
import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import M_jup, R_sun, R_jup, M_earth, R_earth
from platon.visualizer import Visualizer

# All quantities in SI

Rs = 0.947 * R_sun
Mp = 8.145 * M_earth
Rp = 1.7823 * R_earth

T = 1970
logZ = 1.09
CO_ratio = 1.57
log_cloudtop_P = 4.875

#create a TransitDepthCalculator object and compute wavelength dependent transit depths
depth_calculator = TransitDepthCalculator()
wavelengths, transit_depths, info = depth_calculator.compute_depths(
    Rs, Mp, Rp, T, T_star=5196, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=10.0**log_cloudtop_P, full_output=True)

color_bins = 1e-6 * np.array([
    [4, 5],
    [3.2, 4],
    [1.1, 1.7],
])

visualizer = Visualizer()
image, scale = visualizer.draw(info, color_bins, star_color=[1, 1, 0.9],
                               method='disk', star_radius=Rs, max_dist = 7*Rp)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('off')
plt.gca().invert_yaxis()
plt.show()
    
