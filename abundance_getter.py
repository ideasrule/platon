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
        self.metallicities = None
        self.CO_ratios = None
        self.abundances = None
        self.species_set = set()
        self.min_temperature = None
        
        self.load_ggchem_files(include_condensates)


    def load_ggchem_files(self, include_condensates):
        all_logZ = np.linspace(-1, 3, 81)
        self.metallicities = 10**all_logZ
        self.CO_ratios = np.arange(0.2, 2.2, 0.2)
        self.min_temperature = 300
        
        if include_condensates:
            sub_dir = "cond"
        else:
            sub_dir = "gas_only"

        self.abundances = np.load("abundances/{}/all_data.npy".format(sub_dir))
        self.abundances[np.isnan(self.abundances)] = -1
        self.all_species = np.loadtxt("abundances/{}/all_species".format(sub_dir), dtype=str)
        
    def get(self, metallicity, CO_ratio=0.53):
        N_P, N_T, N_species, N_CO, N_Z = self.abundances.shape
        
        reshaped_abundances = self.abundances.reshape((-1, N_CO, N_Z))
        interp_abundances = fast_interpolate(reshaped_abundances, self.metallicities, self.CO_ratios, [metallicity], [CO_ratio])
        interp_abundances = interp_abundances.reshape((N_P, N_T, N_species))
        
        abund_dict = {}
        for i, s in enumerate(self.all_species):
            abund_dict[s] = interp_abundances[:,:,i]

        return abund_dict

    def is_in_bounds(self, metallicity, CO_ratio, T):
        if T <= self.min_temperature: return False
        if metallicity <= np.min(self.metallicities) or metallicity >= np.max(self.metallicities): return False
        if CO_ratio <= np.min(self.CO_ratios) or CO_ratio >= np.max(self.CO_ratios): return False
        return True
    
    def get_metallicity_bounds(self):
        return np.min(self.metallicities), np.max(self.metallicities)

    def get_min_temperature(self):
        return self.min_temperature
