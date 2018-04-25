import sys
import numpy as np
import time
import scipy.linalg
import matplotlib.pyplot as plt

def get_dl(heights, Nh):
    h_mesh, h_prime_mesh = np.meshgrid(heights, heights)

    sqr_length = h_prime_mesh[:, 0:-1]**2 - h_mesh[:, 1:]**2
    sqr_length[sqr_length < 0] = 0
    lengths = 2*np.sqrt(sqr_length)

    dl = lengths[0:-1] - lengths[1:]
    return dl[0:Nh, 0:Nh]

NLam = 4616
Nh = 334

heights = np.loadtxt(sys.argv[1])
assert(len(heights) == Nh + 1)

kappa = np.loadtxt(sys.argv[2])
tau_ref = np.loadtxt(sys.argv[3])

start = time.time()


dl = get_dl(heights, Nh)


result = np.dot(kappa, dl)
print time.time() - start
print np.allclose(result, tau_ref)
