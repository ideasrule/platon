from . import _cupy_numpy as xp
from cupyx.scipy.interpolate import RegularGridInterpolator

def get_condition_array(target_data, interp_data, max_cutoff=xp.inf):
    cond = xp.zeros(len(interp_data), dtype=bool)

    start_index = None
    end_index = None

    for i in range(len(cond)):
        if start_index is None:
            if interp_data[i] > target_data.min():
                start_index = max(0, i-1)
            if interp_data[i] == target_data.min():
                start_index = i
        if end_index is None:
            if interp_data[i] >= target_data.max() or \
               interp_data[i] >= max_cutoff:
                end_index = i + 1
                
    cond[start_index : end_index] = True
    return cond

def interp1d(target_xs, xs, data, assume_sorted=True):
    isscalar = xp.isscalar(target_xs)
    target_xs = xp.atleast_1d(target_xs)
    assert(xs.shape[0] == data.shape[0])
    if not assume_sorted:
        sort = xp.argsort(xs)
        xs = xs[sort]
        data = data[sort]
    
    x_indices = xp.interp(target_xs, xs, xp.arange(len(xs)))
    x_indices_lower = xp.floor(x_indices).astype(int)
    x_indices_upper = xp.ceil(x_indices).astype(int)
    x_indices_frac = x_indices - x_indices_lower

    if not xp.isscalar(target_xs):
        x_indices_frac = x_indices_frac[:,xp.newaxis]

    result = data[x_indices_lower] * (1 - x_indices_frac) + data[x_indices_upper] * x_indices_frac

    if isscalar:
        return result[0]
    
    return result

def regular_grid_interp(ys, xs, data, target_ys, target_xs):
    '''interpolator = RegularGridInterpolator((ys, xs), data, bounds_error=False)
    interpolated_value = interpolator((target_ys, target_xs))
    return interpolated_value'''

    isscalar = xp.isscalar(target_ys)
    target_ys = xp.atleast_1d(target_ys)
    target_xs = xp.atleast_1d(target_xs)
    
    assert(data.shape[0] == len(ys) and data.shape[1] == len(xs))
    assert(len(target_ys) == len(target_xs))
    
    x_indices = xp.interp(target_xs, xs, xp.arange(len(xs)))
    y_indices = xp.interp(target_ys, ys, xp.arange(len(ys)))
        
    x_indices_lower = xp.floor(x_indices).astype(int)
    x_indices_upper = xp.ceil(x_indices).astype(int)
    x_indices_frac = x_indices - x_indices_lower
        
    y_indices_lower = xp.floor(y_indices).astype(int)
    y_indices_upper = xp.ceil(y_indices).astype(int)
    y_indices_frac = y_indices - y_indices_lower

    if data.ndim > 2 and not xp.isscalar(target_ys):
        x_indices_frac = x_indices_frac[:, xp.newaxis]
        y_indices_frac = y_indices_frac[:, xp.newaxis]
        
    result = data[y_indices_lower, x_indices_lower]*(1-y_indices_frac)*(1-x_indices_frac) + \
             data[y_indices_upper, x_indices_lower]*y_indices_frac*(1-x_indices_frac) + \
             data[y_indices_lower, x_indices_upper]*(1-y_indices_frac)*x_indices_frac + \
             data[y_indices_upper, x_indices_upper]*y_indices_frac*x_indices_frac
    if isscalar:
        return result[0]
    
    return result
