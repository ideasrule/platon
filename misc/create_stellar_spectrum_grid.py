import numpy as np
import astropy.io.fits
import sys
import matplotlib.pyplot as plt
import os
import pickle
import astropy.units as u
from astropy.constants import h, c


def air_to_vac(wavelength):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006. Taken from specutils.
    """
    wlum = wavelength.to(u.um).value
    return (1 + 1e-6*(287.6155 + 1.62887/wlum**2 + 0.01360/wlum**4)) * wavelength

binned_wavelengths = np.load("../platon/data/wavelengths.npy") * u.meter

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
        
    wavelengths *= u.Angstrom
    wavelengths = air_to_vac(wavelengths)
    spectrum *= (u.erg/u.cm**2/u.s/u.Angstrom)
    
    binned_spectrum = []

    avg_log_interval = np.median(np.diff(np.log10(binned_wavelengths.value)))
    conversion_factor = None

    for i, wavelength in enumerate(binned_wavelengths):
        start = wavelength * 10**(-avg_log_interval/2.0)
        end = wavelength * 10**(avg_log_interval/2.0)

        cond = np.logical_and(wavelengths >= start, wavelengths < end)
        photon_energy = h*c/wavelength
        photon_flux = np.mean(spectrum[cond])/photon_energy * (end - start)

        

        if conversion_factor is None:
            conversion_factor = photon_flux.si.value / photon_flux.value

        binned_spectrum.append(photon_flux.value)
    #print(len(binned_spectrum))
    binned_spectrum = np.array(binned_spectrum) * conversion_factor
    output_spectra[temperature] = np.array(binned_spectrum)
    print(temperature, np.min(binned_spectrum), np.max(binned_spectrum))

pickle.dump(output_spectra, open("stellar_spectra.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    
