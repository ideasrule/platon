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
            return dictionary
        
