import sys
import numpy as np
import time
import scipy.linalg
import matplotlib.pyplot as plt

def get_dl(heights):
    h_mesh, h_prime_mesh = np.meshgrid(heights, heights)

    sqr_length = h_prime_mesh[:, 0:-1]**2 - h_mesh[:, 1:]**2
    sqr_length[sqr_length < 0] = 0
    lengths = 2*np.sqrt(sqr_length)

    dl = lengths[0:-1] - lengths[1:]
    return dl

heights = np.loadtxt(sys.argv[1]) #Heights in descending order, from center
kappa = np.loadtxt(sys.argv[2]) #N_lambda x N_heights opacities
tau_ref = np.loadtxt(sys.argv[3]) #Tau computed by C Eliza

start = time.time()
dl = get_dl(heights)
result = np.dot(kappa, dl)
print time.time() - start

#Does it match result from C ExoTransmit?
print np.allclose(result, tau_ref)
