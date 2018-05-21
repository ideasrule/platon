import numpy as np
import os
import eos_reader
import scipy
from io import open

from compatible_loader import load_dict_from_pickle

class AbundanceGetter:
    def __init__(self, format='ggchem', include_condensates=False):
        self.metallicities = None
        self.abundances = None
        self.species_set = set()
        self.min_temperature = None
        
        if format == 'ggchem':
            self.load_ggchem_files(include_condensates)
        elif format == 'exotransmit':
            self.load_exotransmit_files(include_condensates)
        else:
            assert(False)
            
    def load_exotransmit_files(self, include_condensates):
        self.min_temperature = 100
        self.metallicities = [0.1, 1, 5, 10, 30, 50, 100, 1000]
        self.abundances = []

        if include_condensates:
            suffix = "cond"
        else:
            suffix = "gas"
        
        for m in self.metallicities:
            m_str = str(m).replace('.', 'p')
            filename = "EOS/eos_{0}Xsolar_{1}.dat".format(m_str, suffix)
            self.abundances.append(eos_reader.get_abundances(filename))
            self.species_set.update(self.abundances[-1].keys())

    def load_ggchem_files(self, include_condensates):
        all_logZ = np.linspace(-1, 3, 81)
        self.metallicities = 10**all_logZ
        self.abundances = []

        if include_condensates:
            sub_dir = "cond"
            self.min_temperature = 300
        else:
            sub_dir = "gas_only"
            self.min_temperature = 100
            
        for i,logZ in enumerate(all_logZ):
            filename = "abundances/{}/abund_dict_{:.2f}.pkl".format(sub_dir, logZ)
            abund = load_dict_from_pickle(filename)
            self.abundances.append(abund)
            self.species_set.update(abund.keys())
            

    def interp(self, metallicity):
        result = dict()
        for key in self.species_set:
            grids = [self.abundances[i][key] for i in range(len(self.abundances))]
            interpolator = scipy.interpolate.interp1d(self.metallicities, grids, axis=0)
            result[key] = interpolator(metallicity)
        return result

    def get_metallicity_bounds(self):
        return np.min(self.metallicities), np.max(self.metallicities)

    def get_min_temperature(self):
        return self.min_temperature
