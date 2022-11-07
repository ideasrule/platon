import pickle
from . import _cupy_numpy as xp
from pkg_resources import resource_filename

def load_dict_from_pickle(filename):
    with open(resource_filename(__name__, filename), "rb") as f:
        dictionary = pickle.load(f, encoding="latin1")
        for key in dictionary:
            dictionary[key] = xp.array(dictionary[key])

        return dictionary
    
def load_numpy(filename):
    return xp.load(resource_filename(__name__, filename))
