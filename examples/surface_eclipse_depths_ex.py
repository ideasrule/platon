import numpy as np 
import matplotlib.pyplot as plt

import astropy.constants as const
import astropy.units as u

from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.TP_profile import Profile
from platon.abundance_getter import AbundanceGetter
from platon.surface_calculator import SurfaceCalculator


#define system parameters
T_star = 3600 #Kelvin   
Rs = 0.4144 * const.R_sun.si.value
semi_major_axis = (0.01406 * u.AU).si.value
Rp = 1.331 * const.R_earth.si.value
Mp = (1.9 * const.M_earth).si.value

def get_abundances(dict):
    getter = AbundanceGetter()
    abundances = getter.get(0, 0.53)
    keys = []
    for key in abundances:
        keys += [key]
        for i,arr in enumerate(abundances[key]):
            abundances[key][i] = [1e-99] * len(arr)
    
    for key in abundances:
        if key in dict:
            abundances[key] += dict[key]
    
    return abundances

#define the gases and their respective VMRs to simulate the atmosphere
gases_dict = dict()
gases_dict['CO2'] = 0.01 
gases_dict['CO'] = 1 - 0.01
abundances = get_abundances(gases_dict)
    
Psurf = 10**(5) #define the surface pressure

calc = EclipseDepthCalculator(method="xsec") #initialize eclipse depth calc
bins = np.array([[5, 5.33], [5.33, 5.66], [5.66, 6], [6, 6.33], [6.33, 6.66], [6.66, 7], [7, 7.33],
                 [7.33, 7.66], [7.66, 8], [8, 8.33], [8.33, 8.66], [8.66, 9], [9, 9.33], [9.33, 9.66],
                 [9.66, 10]]) * 1e-6 #define the bins you'd like

#initialize surface calculator
surface_type = 'Basaltic' #choose between Basaltic, Metal-rich, Fe-oxidized, Granitoid, Ultramafic, Feldspathic, Clay, and Ice-rich silicate
surface = SurfaceCalculator(T_star = T_star, R_star = Rs, a = semi_major_axis, R_planet = Rp,
                            surface_type = surface_type,
                           stellar_blackbody = False, path_to_own_stellar_spectrum = None)
surface.calc_initial_spectra()

#define TP profile
p = Profile()
p.set_isothermal(1000)

bin_change = calc.change_wavelength_bins(bins, surface) #change the bins of both the eclipse_depth and surface calculators

wavelengths, depths, info_dict = calc.compute_depths(p, Rs, Mp, Rp, T_star, surface_temperature = 1172.8,
                                                     logZ=None, CO_ratio=None, stellar_blackbody = False, 
                                                     custom_abundances = abundances, 
                                                     surface_pressure = Psurf,
                                                     surface = surface,
                                                     full_output=True,
                                                    )

unbinned_wavelengths = info_dict['unbinned_wavelengths']
unbinned_depths = info_dict['unbinned_eclipse_depths']

#plot model 
f, ax = plt.subplots(1, 1, figsize = (5,4), constrained_layout = True)
ax.plot(unbinned_wavelengths * 1e6, unbinned_depths*1e6, color = 'cornflowerblue', zorder = 1, label = 'unbinned depths')
ax.scatter(wavelengths * 1e6, depths * 1e6, color = 'k', marker = '.', zorder = 1000, label = 'binned depths')
ax.set_xlabel('wavelength [um]')
ax.set_ylabel('depth [ppm]')
ax.legend()
plt.show()
plt.close()