import numpy as np
import sys
import matplotlib.pyplot as plt
import linecache
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.constants import R_sun, R_jup, M_jup
from platon.TP_profile import Profile
import platon._cupy_numpy as xp

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

p = Profile()
p.pressures = xp.asarray(P_profile)
p.temperatures = xp.asarray(T_profile)

calculator = EclipseDepthCalculator()
wavelengths, depths, _ = calculator.compute_depths(p, 0.75 * R_sun, 1.13 * M_jup, 1.13 * R_jup, 5052, logZ=None, CO_ratio=None, custom_abundances=atm_abundances)

plt.semilogx(1e6 * wavelengths, depths)
plt.show()
