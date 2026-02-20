import pickle
from . import _cupy_numpy as xp
from pathlib import Path

def load_dict_from_pickle(filename):
    basedir = Path(__file__).resolve().parent
    with open(basedir / filename, "rb") as f:
        dictionary = pickle.load(f, encoding="latin1")
        for key in dictionary:
            dictionary[key] = xp.array(dictionary[key])

        return dictionary
    
def load_numpy(filename):
    basedir = Path(__file__).resolve().parent
    return xp.load(basedir / filename)
