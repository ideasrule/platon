import numpy as np
import astropy.io.fits
import sys
import matplotlib.pyplot as plt
import os
import pickle
from astropy.constants import h, c

h = h.si.value
c = c.si.value
M_TO_UM = 1e6

def air_to_vac(wavelength):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006. Taken from specutils. wavelength must be in um.
    """    
    return (1 + 1e-6*(287.6155 + 1.62887/wavelength**2 + 0.01360/wavelength**4)) * wavelength

binned_wavelengths = np.load("R10000/wavelengths.npy")
binned_wavelengths = air_to_vac(binned_wavelengths * M_TO_UM) / M_TO_UM

output_spectra = {}

for temperature in np.arange(2000, 12000, 100):
    filename = "bt-settl-agss2009/lte{0:03d}-4.5-0.0a+0.0.BT-Settl.7.dat.txt".format(int(temperature/100))
    alt_filename = "bt-settl-agss2009/lte0{0}-4.5-0.0.BT-Settl.7.dat.txt".format(int(temperature/100))
    
    if os.path.isfile(filename):
        wavelengths, spectrum = np.loadtxt(filename, unpack=True)
    elif os.path.isfile(alt_filename):
        wavelengths, spectrum = np.loadtxt(alt_filename, unpack=True)
    else:
        continue
    wavelengths *= 1e-10 #Angstrom to SI
    spectrum *= 1e7 #u.erg/u.cm**2/u.s/u.Angstrom to SI

    d_ln_wavelengths = np.median(np.diff(np.log(binned_wavelengths)))
    d_wavelengths = binned_wavelengths * d_ln_wavelengths
    photon_energies = h*c/binned_wavelengths
    
    binned_spectrum = np.interp(binned_wavelengths, wavelengths, spectrum)
    binned_spectrum *= d_wavelengths/photon_energies #photon flux

    output_spectra[temperature] = binned_spectrum
    print(temperature, np.min(binned_spectrum), np.max(binned_spectrum))

pickle.dump(output_spectra, open("stellar_spectra.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    
