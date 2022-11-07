import pickle
import cupy as np
from pkg_resources import resource_filename

def load_dict_from_pickle(filename):
    with open(resource_filename(__name__, filename), "rb") as f:
        dictionary = pickle.load(f, encoding="latin1")
        for key in dictionary:
            dictionary[key] = np.array(dictionary[key])

        return dictionary
    
def load_numpy(filename):
    result = np.load(resource_filename(__name__, filename))
    return result
