from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import nestle
import corner

from platon.fit_info import FitInfo
from platon.retriever import Retriever
from platon.constants import R_sun, R_jup, M_jup
from platon.transit_depth_calculator import TransitDepthCalculator

def stis_bins():
    wave_bins = [[293,347], [348,402], [403,457], [458,512], [512,567], [532,629], [629,726], [727,824], [825,922], [922,1019]]
    wave_bins = 1e-9 * np.array(wave_bins)
    return wave_bins


def wfc3_bins():
    wavelengths = 1e-6*np.array([1.119, 1.138, 1.157, 1.175, 1.194, 1.213, 1.232, 1.251, 1.270, 1.288, 1.307, 1.326, 1.345, 1.364, 1.383, 1.401, 1.420, 1.439, 1.458, 1.477, 1.496, 1.515, 1.533, 1.552, 1.571, 1.590, 1.609, 1.628])
    wavelength_bins = [[w-0.0095e-6, w+0.0095e-6] for w in wavelengths]
    return np.array(wavelength_bins)

def spitzer_bins():
    wave_bins = []
    wave_bins.append([3.2, 4.0])
    wave_bins.append([4.0, 5.0])
    return 1e-6*np.array(wave_bins)


bins = np.concatenate([stis_bins(), wfc3_bins(), spitzer_bins()])

R_guess = 1.4 * R_jup
T_guess = 1200


depth_calculator = TransitDepthCalculator()
depth_calculator.change_wavelength_bins(bins)
wavelengths, depths = depth_calculator.compute_depths(1.19*R_sun, 0.73*M_jup, R_guess, T_guess, T_star=6091)
plt.plot(1e6*wavelengths, depths)
plt.xlabel("Wavelength (um)")
plt.ylabel("Transit depth")
plt.show()

