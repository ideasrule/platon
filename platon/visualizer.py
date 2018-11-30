import numpy as np
import scipy.ndimage

class Visualizer:
    def __init__(self, size=1000):
        '''Initializes the visualizer.
        
        Parameters
        ----------
        size : int
            size x size is the size of the image to draw
        '''
        
        self.size = size

        # Cache pixel_dists for annulus drawing
        xv, yv = np.meshgrid(range(size), range(size))
        self.pixel_dists = np.sqrt((xv - 0.5*size)**2 + (yv - 0.5*size)**2)
        

    def _draw_annulus(self, r1, r2, color_intensities, min_radius, max_radius):
        physical_dists = self.m_per_pix * self.pixel_dists
        in_annulus = np.logical_and(physical_dists > r1, physical_dists < r2)
        self.canvas[in_annulus] = np.array(color_intensities)

    def _draw_layer(self, r1, r2, color_intensities, min_radius, max_radius):
        min_y = round((r1 - min_radius)/self.m_per_pix)
        if min_y < 0: min_y = 0
        min_y = int(min_y)

        max_y = round((r2 - min_radius)/self.m_per_pix)
        if max_y > self.size: max_y = self.size
        max_y = int(max_y)
        
        self.canvas[min_y : max_y, :] = np.array(color_intensities)

    def _draw_star(self, Rstar, max_radius, star_color, margin=0.3):
        Rstar_pix = Rstar/self.m_per_pix
        x_center = (Rstar + margin*max_radius)/self.m_per_pix
        y_center = self.size/2
        xv, yv = np.meshgrid(range(self.size), range(self.size))
        pixel_dists = np.sqrt((xv - x_center)**2 + (yv - y_center)**2)
        self.canvas[pixel_dists > Rstar_pix - 1] = 0
        edge = np.logical_and(pixel_dists > Rstar_pix - 1, pixel_dists < Rstar_pix)
        self.canvas[edge] = (Rstar_pix - pixel_dists[edge])[:, np.newaxis] * np.array(star_color)
            
    def draw(self, transit_info, color_bins, star_color=[1, 1, 1],
             method='disk', star_radius=None, star_margin=0.5,
             max_dist=None, blur_std=1):
        '''
        Draws an image of a transiting exoplanet.

        Parameters
        ----------
        transit_info : dict
            the dictionary returned by compute_depths in TransitDepthCalculator
            when full_output = True
        color_bins : array-like, shape (3,2)
            Wavelength bins to use for the R, G, B channels.  For example, if
            color_bins[0] is [3e-6, 4e-6], the red channel will reflect all
            light transmitted through the atmosphere between 3 and 4 microns.
        star_color : array-like, length 3, optional
            R, G, B values of the star light, with [1,1,1] being white
        method : str, optional
            Either 'disk' to draw the entire planetary disk with atmosphere, or
            'layers' to draw a 1D atmospheric profile--essentially an extreme
            zoom-in on the disk.            
        star_radius : float, optional
            Stellar radius, in meters.  If given, the stellar limb will be drawn
        star_margin : float, optional
            Distance from left side of canvas to stellar limb is star_margin * max_dist
        max_dist : float, optional
            Maximum distance from planet center to draw, in meters
        blur_std : float, optional
            STD of Gaussian blur to apply, in pixels

        Returns
        -------
        canvas : array, shape (self.size, self.size, 3)
            The image of the planet.  Can be displayed with plt.imshow()
        '''

        self.canvas = np.zeros((self.size, self.size, 3))
        radii = np.sort(transit_info["radii"])[::-1]
        if max_dist is None:
            max_dist = np.max(radii)
            
        lambda_grid = transit_info["unbinned_wavelengths"]
        absorption_fraction = 1 - np.exp(-transit_info["tau_los"])
        
        if method == 'disk':
            draw_method = self._draw_annulus
            self.m_per_pix = 2.0 * max_dist / self.size
        elif method == 'layers':
            draw_method = self._draw_layer
            self.m_per_pix = (max_dist - min(radii))/self.size
            if star_radius is not None:
                raise ValueError("Cannot draw star when using layers")
        else:
            raise ValueError("Method must be 'disk' or 'layers'")
                
        for i in range(len(radii) - 1):
            color_intensities = []
            for j, (start, end) in enumerate(color_bins):
                lambda_cond = np.logical_and(lambda_grid > start, lambda_grid < end)
                transmitted_light = 1 - absorption_fraction[:,i][lambda_cond]
                rel_intensity = np.mean(transmitted_light)
                color_intensities.append(rel_intensity)

            color_intensities = star_color * np.array(color_intensities)
            draw_method(radii[i+1], radii[i], color_intensities, min(radii), max_dist)
                
        draw_method(max(radii), np.inf, star_color, min(radii), max_dist)

        if star_radius is not None:
            self._draw_star(star_radius, max_dist, star_color, margin=star_margin)
        self.canvas = scipy.ndimage.gaussian_filter(
            self.canvas, [blur_std, blur_std, 0])
        return self.canvas, self.m_per_pix
    
