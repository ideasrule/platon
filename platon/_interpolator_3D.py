import cupy as np
import sys
import time
from . import __dtype__

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

def interp1d(target_xs, xs, data, assume_sorted=True):
    isscalar = np.isscalar(target_xs)
    target_xs = np.atleast_1d(target_xs)
    if not assume_sorted:
        sort = np.argsort(xs)
        xs = xs[sort]
        data = data[sort]
    
    x_indices = np.interp(target_xs, xs, np.arange(len(xs)))
    x_indices_lower = np.floor(x_indices).astype(int)
    x_indices_upper = np.ceil(x_indices).astype(int)
    x_indices_frac = x_indices - x_indices_lower
    #import pdb
    #pdb.set_trace()
    if not np.isscalar(target_xs):
        x_indices_frac = x_indices_frac[:,np.newaxis]

    result = data[x_indices_lower] * (1 - x_indices_frac) + data[x_indices_upper] * x_indices_frac

    if isscalar:
        return result[0]
    
    return result

def regular_grid_interp(ys, xs, data, target_ys, target_xs):
    isscalar = np.isscalar(target_ys)
    target_ys = np.atleast_1d(target_ys)
    target_xs = np.atleast_1d(target_xs)
    
    assert(data.shape[0] == len(ys) and data.shape[1] == len(xs))
    assert(len(target_ys) == len(target_xs))
    
    x_indices = np.interp(target_xs, xs, np.arange(len(xs)))
    y_indices = np.interp(target_ys, ys, np.arange(len(ys)))
        
    x_indices_lower = np.floor(x_indices).astype(int)
    x_indices_upper = np.ceil(x_indices).astype(int)
    x_indices_frac = x_indices - x_indices_lower
        
    y_indices_lower = np.floor(y_indices).astype(int)
    y_indices_upper = np.ceil(y_indices).astype(int)
    y_indices_frac = y_indices - y_indices_lower

    if data.ndim > 2 and not np.isscalar(target_ys):
        x_indices_frac = x_indices_frac[:, np.newaxis]
        y_indices_frac = y_indices_frac[:, np.newaxis]
        
    result = data[y_indices_lower, x_indices_lower]*(1-y_indices_frac)*(1-x_indices_frac) + \
             data[y_indices_upper, x_indices_lower]*y_indices_frac*(1-x_indices_frac) + \
             data[y_indices_lower, x_indices_upper]*(1-y_indices_frac)*x_indices_frac + \
             data[y_indices_upper, x_indices_upper]*y_indices_frac*x_indices_frac
    if isscalar:
        return result[0]
    
    return result
