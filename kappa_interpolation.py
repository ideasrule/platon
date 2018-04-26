import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline
import time

def normal_interpolate(kappa, grid_T, grid_P, atm_T, atm_P):
    all_results = []
    for i in range(NLam):
        interpolator = RectBivariateSpline(grid_T, grid_P, kappa[i], kx=1, ky=1)
        result = interpolator.ev(atm_T, atm_P)
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

    result = kappa[:, T_indices_lower, P_indices_lower]*(1-T_indices_frac)*(1-P_indices_frac) + \
             kappa[:, T_indices_upper, P_indices_lower]*T_indices_frac*(1-P_indices_frac) + \
             kappa[:, T_indices_lower, P_indices_upper]*(1-T_indices_frac)*P_indices_frac + \
             kappa[:, T_indices_upper, P_indices_upper]*T_indices_frac*P_indices_frac
    end = time.time()
    print end-start
    print result.shape
    return result
    #print T_indices, P_indices
    

temperatures = np.arange(100, 3100, 100)
pressures = 10.0 ** np.arange(-4, 9)

NLam = 4616
NTemp = 30
NPressure = 13

kappa = np.loadtxt(sys.argv[1])
atm_T, atm_P = np.loadtxt(sys.argv[2], unpack=True)

kappa = kappa.reshape((NLam, NTemp, NPressure))
start = time.time()

result_slow = normal_interpolate(kappa, temperatures, pressures, atm_T, atm_P)
#print result

result_fast = fast_interpolate(kappa, temperatures, pressures, atm_T, atm_P)
#print result

print np.allclose(result_slow, result_fast)
exit(0)

all_results = []
#scipy.interpolate.RectBivariateSpline(temperatures, pressures, kappa, kx=1, ky=1)
xv, yv = np.meshgrid(np.arange(NTemp), np.arange(NPressure))
print xv.shape
print xv, yv
test = scipy.interpolate.RectBivariateSpline(temperatures, pressures, yv.T, kx=1, ky=1)
y_indices = test(atm_T, atm_P, grid=False)

test = scipy.interpolate.RectBivariateSpline(temperatures, pressures, xv.T, kx=1, ky=1)
x_indices = test.ev(atm_T, atm_P)
print "kappa shape", kappa[:, x_indices.astype(int), y_indices.astype(int)].shape
#print kappa[0][indices.astype(int)].shape

for i in range(NLam):
    interpolator = scipy.interpolate.RectBivariateSpline(temperatures, pressures, kappa[i], kx=1, ky=1)
    result = interpolator.ev(atm_T, atm_P)
    all_results.append(result)

end = time.time()
print end-start
