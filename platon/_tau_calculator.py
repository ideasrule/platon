import sys
import numpy as np
import time
import scipy.linalg
import matplotlib.pyplot as plt


def get_dl(radii):
    '''Radii must be in descending order.  Returns dl[i,j], the distance travelled by a ray with impact parameter radii[j+1] from the shell at radii[i+1] to radii[i].'''

    r_mesh, r_prime_mesh = np.meshgrid(radii, radii)
    sqr_length = r_prime_mesh[:, 0:-1]**2 - r_mesh[:, 1:]**2
    sqr_length[sqr_length < 0] = 0
    lengths = 2 * np.sqrt(sqr_length)

    dl = lengths[0:-1] - lengths[1:]
    return dl


def get_line_of_sight_tau(absorption_coeff, radii):
    '''radii must be in descending order.  absorption_coeff must be an array whose first dimension is the number of wavelengths and second dimension is the number of radius points, which must be 1 less than the size of radii.''',

    assert(np.allclose(radii, np.sort(radii)[::-1]))
    intermediate_coeff = 0.5 * \
        (absorption_coeff[0:-1] + absorption_coeff[1:])
    dl = get_dl(radii)
    return np.dot(intermediate_coeff.T, dl)
