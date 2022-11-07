import pickle
import cupy as np
from pkg_resources import resource_filename
from . import __dtype__

def load_dict_from_pickle(filename):
    with open(resource_filename(__name__, filename), "rb") as f:
        dictionary = pickle.load(f, encoding="latin1")
        for key in dictionary:
            dictionary[key] = np.array(dictionary[key], dtype=__dtype__)

        #import pdb
        #pdb.set_trace()
        return dictionary
    
def load_numpy(filename):
    result = np.array(np.load(resource_filename(__name__, filename)), dtype=__dtype__)
    #import pdb
    #pdb.set_trace()
    return result
