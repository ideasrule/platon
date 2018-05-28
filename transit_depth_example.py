import numpy as np
import matplotlib.pyplot as plt
import corner

from fit_info import FitInfo
import eos_reader
from abundance_getter import AbundanceGetter
from transit_depth_calculator import TransitDepthCalculator
from retrieve import Retriever

depth_calculator = TransitDepthCalculator(7e8, 9.8, max_P_profile=1.014e5)
custom_abundances = eos_reader.get_abundances("EOS/eos_1Xsolar_cond.dat")
wavelengths, transit_depths = depth_calculator.compute_depths(6.4e6, 800, custom_abundances=custom_abundances)

ref_wavelengths, ref_depths = np.loadtxt("testing_data/ref_spectra.dat", unpack=True, skiprows=2)
ref_depths /= 100

plt.plot(ref_wavelengths, ref_depths, label="ExoTransmit")
plt.plot(wavelengths, transit_depths, label="PyExoTransmit")
plt.legend()
plt.show()
