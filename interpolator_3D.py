import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline
import time

def normal_interpolate(data, grid_x, grid_y, target_x, target_y):
    all_results = []
    for i in range(data.shape[0]):
        interpolator = RectBivariateSpline(grid_x, grid_y, data[i], kx=1, ky=1)
        result = interpolator.ev(target_x, target_y)
        all_results.append(result)
    return np.array(all_results)
    

def fast_interpolate(data, grid_x, grid_y, target_x, target_y):
    T_mesh, P_mesh = np.meshgrid(np.arange(len(grid_x)), np.arange(len(grid_y)))
    interpolator = RectBivariateSpline(grid_x, grid_y, T_mesh.T, kx=1, ky=1)
    x_indices = interpolator.ev(target_x, target_y)

    x_indices_lower = x_indices.astype(int)
    x_indices_upper = x_indices_lower + 1
    x_indices_frac = x_indices - x_indices_lower

    interpolator = RectBivariateSpline(grid_x, grid_y, P_mesh.T, kx=1, ky=1)
    y_indices = interpolator.ev(target_x, target_y)
    y_indices_lower = y_indices.astype(int)
    y_indices_upper = y_indices_lower + 1
    y_indices_frac = y_indices - y_indices_lower

    result = data[:, y_indices_lower, x_indices_lower]*(1-y_indices_frac)*(1-x_indices_frac) + \
             data[:, y_indices_upper, x_indices_lower]*y_indices_frac*(1-x_indices_frac) + \
             data[:, y_indices_lower, x_indices_upper]*(1-y_indices_frac)*x_indices_frac + \
             data[:, y_indices_upper, x_indices_upper]*y_indices_frac*x_indices_frac
    return result
    

