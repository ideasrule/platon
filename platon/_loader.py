import pickle
import numpy as np
from pkg_resources import resource_filename

def load_dict_from_pickle(filename):
    with open(resource_filename(__name__, filename), "rb") as f:
        return pickle.load(f, encoding="latin1")
        
def load_numpy(filename):
    return np.load(resource_filename(__name__, filename))
