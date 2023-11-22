import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import ascii
import pandas as pd

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.surface_calculator import SurfaceCalculator 

import astropy.constants as const
import astropy.units as u

#define system parameters 
T_star = 3573 
Rs = (0.4144 * const.R_sun).si.value  
semi_major_axis = (0.01406 * u.AU).si.value 
Mp = (1.9 * const.M_earth ).si.value
Rp = (1.331 * const.R_earth).si.value

surface_type = 'Basaltic' #surface type to consider

path_to_sim_data = f'surface_retrieval_simulated_data.csv'

simulated_data = pd.read_csv(f'{path_to_sim_data}',sep = '\t', header = 0)
eclipse_wavelengths = np.array(simulated_data['wavelengths']) * 1e-6 #m
eclipse_depths = np.array(simulated_data['depths']) 
eclipse_errors = np.array(simulated_data['depth_errors'])


bins = pd.read_csv(f'surface_retrieval_bins.csv', sep = '\t')
bins_combined = []
for a, b in zip(bins['a'].to_numpy(), bins['b'].to_numpy()):
    bins_combined += [[a,b]]
bins_combined = np.array(bins_combined) * 1e-6
bins = bins_combined

#create a Retriever object
retriever = CombinedRetriever()

#initialize surface calculator
surface = SurfaceCalculator(T_star = T_star, R_star = Rs, a = semi_major_axis, R_planet = Rp,
                            surface_type = surface_type,
                           stellar_blackbody = False)#, path_to_own_stellar_spectrum = path_to_own_stellar_spectrum)
surface.calc_initial_spectra()

fit_info = retriever.get_default_fit_info(
    Rs=Rs, Mp=Mp, Rp=Rp, logZ = None, CO_ratio = None,
    custom_abundances = None, stellar_blackbody = False, 
    T_star=T_star, a = semi_major_axis, surface = surface, 
    T_surf = 1400, use_clr = True, f = None, log_surface_P = 3, nircam_offset = None,
    gases = ['CO2', 'N2'], clr_CO2 = 2, #the last gas in gases is the background gas
    path_to_own_stellar_spectrum = None,
    T = 1000, profile_type="isothermal" 
    )

fit_info.add_uniform_fit_param("T", 500, 2000)

fit_info.add_uniform_fit_param("log_surface_P", 0, 8)

fit_info.add_uniform_fit_param("T_surf", 500, 2000)# can define a Tsurf, if not will use the T at the bottom of the TP profile

fit_info.add_CLR_fit_param("clr_CO2", 1) #for CLR priors, just put the number of gases considering (not including the background gas)

result = retriever.run_multinest(None, None, None,
                                 bins, eclipse_depths, eclipse_errors,
                                 fit_info, nlive=500,
                                 sample="rwalk",
                                 rad_method="xsec", surface = surface)

with open(f"result.pkl", "wb") as f:
    pickle.dump(result, f)

#Plot the spectrum and save it to best_fit.png
result.plot_spectrum(f"best_fit", true_model = None)

#Plot the 2D posteriors with "corner" package and save it to multinest_corner.png
result.plot_corner(filename = f"corner.png",
                   file_name_vmr = f"corner_vmr.png",
                   truths = None)

