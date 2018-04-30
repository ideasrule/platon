import sys
import numpy as np
import time
import scipy.linalg
import matplotlib.pyplot as plt

def get_dl(heights):
    '''Given heights above the planet center (not the surface) in descending
    order, returns dl(h, h').  This is the distance that a ray with impact
    parameter h travels to get from height h' to the next height (namely the
    next lowest height in the array).'''
    h_mesh, h_prime_mesh = np.meshgrid(heights, heights)

    sqr_length = h_prime_mesh[:, 0:-1]**2 - h_mesh[:, 1:]**2
    sqr_length[sqr_length < 0] = 0
    lengths = 2*np.sqrt(sqr_length)

    dl = lengths[0:-1] - lengths[1:]
    return dl

def get_line_of_sight_tau(absorption_coeff, heights):
    #heights must be in descending order
    assert(np.allclose(heights, np.sort(heights)[::-1]))
    
    dl = get_dl(heights)
    return np.dot(absorption_coeff, dl)
