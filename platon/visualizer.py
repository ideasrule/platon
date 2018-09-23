import numpy as np
import matplotlib.pyplot as plt

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import M_jup, R_sun, R_jup, M_earth, R_earth


# Structure

# Class: Visualizer
  #Initialize: scale, center, size; set up blank canvas
# draw: take data and 3 wavelength ranges to draw
# draw_annulus: draws onto canvas

class Visualizer:
    def __init__(self, size=1000):
        self.size = size
        self.canvas = np.zeros((size, size, 3))

        # Cache pixel_dists for annulus drawing
        xv, yv = np.meshgrid(range(size), range(size))
        self.pixel_dists = np.sqrt((xv - 0.5*size)**2 + (yv - 0.5*size)**2)
        

    def draw_annulus(self, r1, r2, color_intensities, min_radius, max_radius):
        m_per_pix = 2.0 * max_radius / self.size
        physical_dists = m_per_pix * self.pixel_dists
        in_annulus = np.logical_and(physical_dists > r1, physical_dists < r2)
        self.canvas[in_annulus] = np.array(color_intensities)

    def draw_layer(self, r1, r2, color_intensities, min_radius, max_radius):
        scale = (max_radius - min_radius)/self.size
        min_y = round((r1 - min_radius)/scale)
        if min_y < 0: min_y = 0
        min_y = int(min_y)

        max_y = round((r2 - min_radius)/scale)
        if max_y > self.size: max_y = self.size
        max_y = int(max_y)
        
        #print(min_y, max_y, scale)
        self.canvas[min_y : max_y, :] = np.array(color_intensities)

            
    def draw(self, transit_info, color_bins, color_mult_factors=[1, 1, 1], method='disk'):
        #absorption_fraction: Nlambda x Nradii
        #stellar_spectrum: NLambda
        #color_bins: 3x2, specifying (start, end) for RGB

        radii = np.sort(transit_info["radii"])[::-1]
        lambda_grid = transit_info["unbinned_wavelengths"]
        absorption_fraction = 1 - np.exp(-transit_info["tau_los"])
        
        if method == 'disk':
            draw_method = self.draw_annulus
        elif method == 'layers':
            draw_method = self.draw_layer
        else:
            raise ValueError("Method must be 'disk' or 'layers'")
                
        for i in range(len(radii) - 1):
            color_intensities = []
            for j, (start, end) in enumerate(color_bins):
                lambda_cond = np.logical_and(lambda_grid > start, lambda_grid < end)
                transmitted_light = 1 - absorption_fraction[:,i][lambda_cond]
                #if j == 2:
                #    transmitted_light = 1.5*transmitted_light**0.5/(0.5 + transmitted_light**0.5)
                rel_intensity = np.mean(transmitted_light * color_mult_factors[j])
                
                color_intensities.append(rel_intensity)
            #print(radii[i], radii[i+1], color_intensities)
            color_intensities = np.array(color_intensities) #* np.array(color_mult_factors)
            #print color_intensities
            draw_method(radii[i+1], radii[i], color_intensities, np.min(radii), np.max(radii))
                
        draw_method(radii[0], np.inf, color_mult_factors, np.min(radii), np.max(radii))
        return self.canvas
    

# All quantities in SI

#samples = np.load("samples.npy")
#logl = np.load("logl.npy")

Rs = 0.947 * R_sun
Mp = 8.145 * M_earth
Rp = 1.7823 * R_earth

T = 1970
logZ = 1.09
CO_ratio = 1.57
log_cloudtop_P = 4.875

#Rs, Mp, Rp, T, logZ, CO_ratio, log_cloudtop_P, error_multiple = samples[np.argmax(logl)]

print Rs/R_sun, Mp/M_earth, Rp/R_earth, T, logZ, CO_ratio, log_cloudtop_P

#create a TransitDepthCalculator object and compute wavelength dependent transit depths
depth_calculator = TransitDepthCalculator()
wavelengths, transit_depths, info = depth_calculator.compute_depths(
    Rs, Mp, Rp, T, T_star=5196, logZ=logZ, CO_ratio=CO_ratio, cloudtop_pressure=10.0**log_cloudtop_P, full_output=True)

#plt.plot(1e6*wavelengths, transit_depths)
#plt.xlim(4, 5)
#plt.show()

color_bins = 1e-6 * np.array([
    [4, 5],
    [3.2, 4],
    [1.1, 1.7],
])

#color_bins = 1e-6 * np.array([
#[1.5, 1.7],
#[1.4, 1.5],
#[1.1, 1.4]])

'''color_bins = 1e-6 * np.array([
[0.5, 0.7],
[0.43, 0.67],
[0.38, 0.55]
])'''

visualizer = Visualizer()
image = visualizer.draw(info, color_bins, color_mult_factors=[1, 1, 0.9], method='layers')

plt.imshow(image)
plt.axis('off')
plt.gca().invert_yaxis()
plt.show()
    
