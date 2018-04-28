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

def get_line_of_sight_tau(absorption_coeff, heights):
    dl = get_dl(heights)
    return np.dot(absorption_coeff, dl)
