import numpy as np
import os
import eos_reader
import scipy
from io import open
import time

from interpolator_3D import fast_interpolate
from compatible_loader import load_dict_from_pickle

class AbundanceGetter:
    def __init__(self, include_condensates=False):
        self.min_temperature = 300
        self.logZs = np.linspace(-1, 3, 81)
        self.CO_ratios = np.arange(0.2, 2.2, 0.2)
        
        if include_condensates:
            sub_dir = "cond"
        else:
            sub_dir = "gas_only"

        self.log_abundances = np.log10(np.load("abundances/{}/all_data.npy".format(sub_dir)))
        self.included_species = np.loadtxt("abundances/{}/included_species".format(sub_dir), dtype=str)

        
    def get(self, logZ, CO_ratio=0.53):
        N_P, N_T, N_species, N_CO, N_Z = self.log_abundances.shape
        
        reshaped_log_abund = self.log_abundances.reshape((-1, N_CO, N_Z))
        interp_log_abund = 10 ** fast_interpolate(
            reshaped_log_abund, self.logZs, np.log10(self.CO_ratios),
            logZ, np.log10(CO_ratio))
        interp_log_abund = interp_log_abund.reshape((N_P, N_T, N_species))
        
        abund_dict = {}
        for i, s in enumerate(self.included_species):
            abund_dict[s] = interp_log_abund[:,:,i]

        return abund_dict

    
    def is_in_bounds(self, logZ, CO_ratio, T):
        if T <= self.min_temperature: return False
        if logZ <= np.min(self.logZs) or logZ >= np.max(self.logZs): return False
        if CO_ratio <= np.min(self.CO_ratios) or CO_ratio >= np.max(self.CO_ratios): return False
        return True
    
