import numpy as np
import astropy.io.fits
import sys
import matplotlib.pyplot as plt
import os
import pickle

binned_wavelengths = np.load("../platon/data/wavelengths.npy") * 1e10
log_wavelengths = np.log10(binned_wavelengths)

output_spectra = {}

#wavelengths, spectrum = np.loadtxt(sys.argv[1], unpack=True)
for temperature in np.arange(2000, 12000, 100):
    filename = "bt-nextgen-agss2009/lte{0:03d}-4.5-0.0a+0.0.BT-NextGen.7.dat.txt".format(temperature/100)
    alt_filename = "bt-nextgen-agss2009/lte0{0}-4.5-0.0.BT-NextGen.7.dat.txt".format(temperature/100)
    
    if os.path.isfile(filename):
        wavelengths, spectrum = np.loadtxt(filename, unpack=True)
    elif os.path.isfile(alt_filename):
        wavelengths, spectrum = np.loadtxt(alt_filename, unpack=True)
    else:
        continue
    
    binned_spectrum = []

    avg_log_interval = np.median(log_wavelengths[1:] - log_wavelengths[0:-1])
    for i, log_w in enumerate(log_wavelengths):
        start = 10**(log_w - avg_log_interval/2.0)
        end = 10**(log_w + avg_log_interval/2.0)

        cond = np.logical_and(wavelengths >= start, wavelengths < end)
        binned_spectrum.append(np.mean(spectrum[cond]))
        
    output_spectra[temperature] = np.array(binned_spectrum)
    print temperature

pickle.dump(output_spectra, open("stellar_spectra.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    
