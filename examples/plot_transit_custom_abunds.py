import numpy as np
import sys
import matplotlib.pyplot as plt
import linecache
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup

bar_to_cgs = 1e6
Pa_to_cgs = 10

data_file = "example_custom_abundances.txt"

data = np.loadtxt(data_file, skiprows=2)[::-1]
P_profile = data[:,0] / Pa_to_cgs
T_profile = data[:,1]
T_profile[T_profile > 3000] = 3000

#plt.semilogy(T_profile, P_profile)
#plt.show()

header = linecache.getline(data_file, 2).split()
included_species = ["CO", "CO2", "C2H2", "H2", "H", "H2O", "HCN", "He", "NH3", "O2", "NO", "OH"]
atm_abundances = {}

for i, s in enumerate(included_species):
    index = header.index(s)
    atm_abundances[s] = data[:, index] 
    #plt.loglog(P_profile, atm_abundances[s], label=s)

#plt.legend()
#plt.show()

calculator = TransitDepthCalculator()
wavelengths, depths = calculator.compute_depths(0.75 * R_sun, 1.13 * M_jup, 1.13 * R_jup, None, atm_abundances, custom_T_profile=T_profile, custom_P_profile=P_profile)
np.save("wavelengths.npy", wavelengths)
np.save("without_ch4_depths.npy", depths)
plt.semilogx(1e6 * wavelengths, depths)
#plt.ylim(0.0228, 0.0256)
plt.show()
