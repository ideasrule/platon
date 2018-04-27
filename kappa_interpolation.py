import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline
import time

def normal_interpolate(kappa, grid_T, grid_P, atm_T, atm_P):
    all_results = []
    N_wavelengths = kappa.shape[0]
    for i in range(N_wavelengths):
        interpolator = RectBivariateSpline(grid_P, grid_T, kappa[i], kx=1, ky=1)
        result = interpolator.ev(atm_P, atm_T)
        all_results.append(result)
    return np.array(all_results)
    

def fast_interpolate(kappa, grid_T, grid_P, atm_T, atm_P):
    start = time.time()
    T_mesh, P_mesh = np.meshgrid(np.arange(len(grid_T)), np.arange(len(grid_P)))
    interpolator = RectBivariateSpline(grid_T, grid_P, T_mesh.T, kx=1, ky=1)
    T_indices = interpolator.ev(atm_T, atm_P)

    T_indices_lower = T_indices.astype(int)
    T_indices_upper = T_indices_lower + 1
    T_indices_frac = T_indices - T_indices_lower

    interpolator = RectBivariateSpline(grid_T, grid_P, P_mesh.T, kx=1, ky=1)
    P_indices = interpolator.ev(atm_T, atm_P)
    P_indices_lower = P_indices.astype(int)
    P_indices_upper = P_indices_lower + 1
    P_indices_frac = P_indices - P_indices_lower

    result = kappa[:, P_indices_lower, T_indices_lower]*(1-P_indices_frac)*(1-T_indices_frac) + \
             kappa[:, P_indices_upper, T_indices_lower]*P_indices_frac*(1-T_indices_frac) + \
             kappa[:, P_indices_lower, T_indices_upper]*(1-P_indices_frac)*T_indices_frac + \
             kappa[:, P_indices_upper, T_indices_upper]*P_indices_frac*T_indices_frac
    end = time.time()
    print end-start
    return result
    

'''temperatures = np.arange(100, 3100, 100)
pressures = 10.0 ** np.arange(-4, 9)

N_wavelengths = 4616
N_temperatures = 30
N_pressures = 13

kappa = np.loadtxt(sys.argv[1])
atm_T, atm_P = np.loadtxt(sys.argv[2], unpack=True)

kappa = kappa.reshape((N_wavelengths, N_temperatures, N_pressures))

result_slow = normal_interpolate(kappa, temperatures, pressures, atm_T, atm_P)
result_fast = fast_interpolate(kappa, temperatures, pressures, atm_T, atm_P)
print np.allclose(result_slow, result_fast)'''
