import numpy as np 
import matplotlib.pyplot as plt

from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.TP_profile import Profile
from platon.abundance_getter import AbundanceGetter
from platon.surface_calculator import SurfaceCalculator
from platon.constants import R_sun, AU, R_earth, M_earth


#define system parameters
T_star = 3600
Rs = 0.4144 * R_sun
semi_major_axis = 0.01406 * AU
Rp = 1.331 * R_earth
Mp = 1.9 * M_earth
Psurf = 1e5 #1 bar
surface_library="HES2012" #Or Paragas
surface_type = "Basaltic"
surface_temp = 1172.8

#define the gases and their respective VMRs to simulate the atmosphere
abundances = AbundanceGetter().get(0, 0.5)
for key in abundances:
    abundances[key] *= 0
abundances["CO2"] += 0.01
abundances["CO"] += 1 - 0.01

calc = EclipseDepthCalculator(surface_library=surface_library)
bins = np.array([[5, 5.33], [5.33, 5.66], [5.66, 6], [6, 6.33], [6.33, 6.66], [6.66, 7], [7, 7.33],
                 [7.33, 7.66], [7.66, 8], [8, 8.33], [8.33, 8.66], [8.66, 9], [9, 9.33], [9.33, 9.66],
                 [9.66, 10]]) * 1e-6 
calc.change_wavelength_bins(bins)

#define TP profile
p = Profile()
p.set_isothermal(1000)

wavelengths, depths, info_dict = calc.compute_depths(
    p, Rs, Mp, Rp, T_star, 
    logZ=None, CO_ratio=None, stellar_blackbody = False, 
    custom_abundances = abundances, 
    surface_pressure = Psurf,
    surface_type=surface_type, semimajor_axis=semi_major_axis, surface_temp = surface_temp,
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
