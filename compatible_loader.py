import pickle
import numpy as np
import copy

def load_dict_from_pickle(filename):
    with open(filename, "rb") as f:    
        try:
            # Python 2
            return pickle.load(f)
        except UnicodeDecodeError:
            # Python 3
            f.seek(0)
            dictionary = pickle.load(f, encoding="latin1")
            original_keys = list(dictionary.keys())
            
            for key in original_keys:
                dictionary[key.decode('utf-8')] = dictionary[key]
                del dictionary[key]

            return dictionary

def load_numpy_array(filename):
    try:
        # Python 2
        return np.load(filename)
    except OSError:
        # Python 3
        return np.load(filename, encoding='latin1')
