import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline, interp1d
import time


def get_condition_array(target_data, interp_data, max_cutoff=np.inf):
    cond = np.zeros(len(interp_data), dtype=bool)

    start_index = None
    end_index = None

    for i in range(len(cond)):
        if start_index is None:
            if interp_data[i] > np.min(target_data):
                start_index = i-1
            if interp_data[i] == np.min(target_data):
                start_index = i
        if end_index is None:
            if interp_data[i] >= np.max(target_data) or \
               interp_data[i] >= max_cutoff:
                end_index = i + 1
                
    cond[start_index : end_index] = True
    return cond


