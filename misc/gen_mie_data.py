#!/home/zmzhang/miniconda3/bin/python -u
#SBATCH -c 1
#SBATCH --mem 190G
#SBATCH -t 7-0

'''Get absorption cross sections for a log-normal radius distribution of Mie scattering particles with various compositions.  Uses wavelength-dependent complex refractive indices.'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os.path
import scipy.ndimage
from platon._mie_multi_x import get_Qext

M_TO_MICRON = 1e6
radii_log_spacing = 1/20
mie_radii = np.exp(np.arange(np.log(5e-11), np.log(2e-4), radii_log_spacing))
np.save("mie_radii.npy", mie_radii)

grid_wavelengths = np.load("low_res_lambdas.npy")
grid_wavelengths[0] += 1e-22 #Hack to get around the fact that many refractive index data files start at exactly 0.2 um

def get_xsec_grid(filename):
    wavelengths, n, k = np.loadtxt(filename, unpack=True)
    wavelengths /= M_TO_MICRON
    
    xsecs = []
    for i in range(len(wavelengths)):
        xs = 2*np.pi*mie_radii / wavelengths[i]
        if wavelengths[i] < grid_wavelengths[0] or wavelengths[i] > grid_wavelengths[-1]:
            xsecs.append(np.zeros(len(mie_radii)))
        else:
            xsecs.append(np.pi * mie_radii**2 * get_Qext(n[i] - k[i]*1j, xs))
        print(filename, i, len(wavelengths))

    xsecs = scipy.interpolate.interp1d(wavelengths, xsecs, axis=0)(grid_wavelengths)   
    return xsecs


f = open("all_cross_secs.pkl", "wb")
output_dict = {}
for filename in sys.argv[1:]:
    print(filename)
    species_name = os.path.basename(filename).replace(".dat", "")
    if species_name in ["Fe2SiO4", "MgAl2O4"]:
        continue
    temp_filename = "unsmoothed_{}.npy".format(species_name)
    if os.path.exists(temp_filename):
        output_dict[species_name] = np.load(temp_filename)
    else:
        output_dict[species_name] = get_xsec_grid(filename)
        np.save(temp_filename, output_dict[species_name])
    
pickle.dump(output_dict, f)
f.close()
