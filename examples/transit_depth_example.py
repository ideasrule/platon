import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile
from platon.constants import M_jup, R_sun, R_jup

# All quantities in SI
Rs = 1.16 * R_sun     #Radius of star
Mp = 0.73 * M_jup     #Mass of planet
Rp = 1.40 * R_jup      #Radius of planet
T = 1200              #Temperature of isothermal part of the atmosphere
M_TO_UM = 1e6
PA_TO_BAR = 1e-5

p = Profile.isothermal(T)
depth_calculator = TransitDepthCalculator()
wavelengths, transit_depths, info_dict = depth_calculator.compute_depths(
    p, Rs, Mp, Rp, logZ=0, CO_ratio=0.5, cloudtop_pressure=1e4, full_output=True)

plt.pcolormesh(M_TO_UM * wavelengths, PA_TO_BAR * info_dict["P_profile"],
               info_dict["contrib"].T, shading="nearest", antialiased=True)
plt.xscale("log")
plt.xticks([0.3, 0.5, 1, 2, 3, 5, 10, 20])
plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%g"))
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.yscale("log")
plt.gca().invert_yaxis()
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Pressure (bar)")
plt.figure()

plt.semilogx(1e6*wavelengths, transit_depths)
plt.xlabel("Wavelength (um)")
plt.ylabel("Transit depth")
plt.show()
