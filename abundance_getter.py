import numpy as np
import os
import eos_reader
import scipy
import pickle

class AbundanceGetter:
    def __init__(self, format='ggchem'):
        self.metallicities = None
        self.abundances = None
        
        if format == 'ggchem':
            self.load_ggchem_files()
        elif format == 'exotransmit':
            self.load_exotransmit_files()
        else:
            assert(False)
            
    def load_exotransmit_files(self, type='gas'):
        self.metallicities = [0.1, 1, 5, 10, 30, 50, 100, 1000]
        self.abundances = []
        for m in self.metallicities:
            m_str = str(m).replace('.', 'p')
            filename = "EOS/eos_{0}Xsolar_{1}.dat".format(m_str, type)
            print filename
            self.abundances.append(eos_reader.get_abundances(filename))

    def load_ggchem_files(self):
        all_logZ = np.linspace(-1, 3, 81)
        self.metallicities = 10**all_logZ
        self.abundances = []
        file_exists = np.ones(len(all_logZ), dtype=bool)
        
        for i,logZ in enumerate(all_logZ):
            filename = "abundances/abund_dict_{0}.pkl".format(str(logZ))
            if not os.path.isfile(filename):
                file_exists[i] = False
                continue

            with open(filename) as f:                
                self.abundances.append(pickle.load(f))
            
        self.metallicities = self.metallicities[file_exists]

    def interp(self, metallicity):
        result = dict()
        for key in self.abundances[0]:
            grids = [self.abundances[i][key] for i in range(len(self.abundances))]
            interpolator = scipy.interpolate.interp1d(self.metallicities, grids, axis=0)
            result[key] = interpolator(metallicity)
        return result

    def get_metallicity_bounds(self):
        return np.min(self.metallicities), np.max(self.metallicities)
