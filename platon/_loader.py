import pickle
import numpy as np
from pathlib import Path


def load_dict_from_pickle(filename):
    basedir = Path(__file__).resolve().parent
    with open(basedir / filename, "rb") as f:
        dictionary = pickle.load(f, encoding="latin1")
        for key in dictionary:
            dictionary[key] = np.asarray(dictionary[key])

        return dictionary


def load_numpy(filename):
    basedir = Path(__file__).resolve().parent
    return np.load(basedir / filename)
