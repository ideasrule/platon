import numpy as np
import scipy
import scipy.special as spl
import copy
import pdb

def get_Jn_log_deriv(x, refractive_index, num, ru):
    s = 1.0 / (x * refractive_index)
    wave_indices = np.arange(len(x))
    ru[[num-1],[wave_indices]] = num * s
    #ru[num-1] = (num) * s #try (num + 1)*s instead
    counter = copy.deepcopy(num)
    while np.any(np.greater_equal(counter, 2)):
        wave_indices = wave_indices[counter>=2]
        counter = counter[counter>=2]
        s1 = counter * s[wave_indices]
        ru[[counter-2],[wave_indices]] = s1 - 1.0/ (ru[[counter-1],[wave_indices]] + s1 )
        counter = counter - 1
    return ru

def get_iterations_required(refractive_index, x):
    y = np.sqrt(np.real(refractive_index * np.conj(refractive_index))) * x
    num = 1.25 * y + 15.5

    num[y<1.0] = 7.5 * y[y<1.0] + 9.0

    num[(y>100.0) & (y<50000.0)] = 1.0625 * y[(y>100.0) & (y<50000.0)] + 28.5

    num[y>50000.0] = 1.005 * y[y>50000.0] + 50.5

    num = num.astype(int)
    return num

def get_Qext(refractive_index, x, eps=1e-15, xmin=1e-6):
    Qext    = np.zeros(len(x))

    fact = np.array([1.0,1.0e+150])
    factor = 1.0e+150

    if np.any(np.less_equal(x,xmin)):
        raise ValueError("Mie size parameter {} less than minimum {}".format(x, xmin))

    num_iterations = get_iterations_required(refractive_index, x)
    ru = np.zeros((np.max(num_iterations), len(x)), dtype=complex)

    ru = get_Jn_log_deriv(x, refractive_index, num_iterations, ru)

    ass = np.sqrt(np.pi / 2.0 / x)
    w1 = 2.0 / np.pi / x
    Si = np.sin(x) / x
    Co = np.cos(x) / x

    besJ0 = np.sin(x) * np.sqrt(2/np.pi/x)
    besY0 = - np.cos(x) * np.sqrt(2/np.pi/x)
    iu0 = 0

    besJ1 = ( Si / x - Co) / ass
    besY1 = (-Co / x - Si) / ass
    iu1 = 0
    iu2 = 0

    #Mie Coefficients
    s = ru[0] / refractive_index + 1.0/x
    s1 = s * besJ1 - besJ0
    s2 = s * besY1 - besY0
    ra0 = s1 / (s1 + 1j * s2)

    s   = ru[0] * refractive_index + 1.0/x
    s1  = s * besJ1 - besJ0
    s2  = s * besY1 - besY0
    rb0 = s1 / (s1 + 1j * s2)

    Qext = np.real(3 * (ra0 + rb0))

    for i in range(1, np.max(num_iterations)):
        indices = np.less(i,num_iterations)
        besY2 = np.zeros(len(x))
        besJ2 = np.zeros(len(x))

        if iu1 == iu0:
            besY2[indices] = (2 * i + 1) / x[indices] * besY1[indices] - besY0[indices]
        else:
            besY2[indices] = (2 * i + 1) / x[indices] * besY1[indices] - besY0[indices] / factor

        if np.any(np.greater(np.abs(besY2),1.0e+200)):
            besY2[np.abs(besY2) > 1.0e+200] = besY2[np.abs(besY2) > 1.0e+200] / factor
            iu2 = iu1 + 1

        besJ2[indices] = (w1[indices] + besY2[indices] * besJ1[indices]) / besY1[indices]
        s = ru[i,indices] / refractive_index + (i + 1) / x[indices]
        if iu1 > 1 or iu2 > 1:
            raise ValueError("iu1 > 1 or iu2 > 1")

        s1 = s * besJ2[indices] / fact[iu2] - besJ1[indices] / fact[iu1]
        s2 = s * besY2[indices] / fact[iu2] - besY1[indices] / fact[iu1]
        ra1 = s1 / (s1 + 1j * s2)

        s = ru[i,indices] * refractive_index + (i + 1) * 1.0/x[indices]
        s1 = s * besJ2[indices] / fact[iu2] - besJ1[indices] / fact[iu1]
        s2 = s * besY2[indices] / fact[iu2] - besY1[indices] / fact[iu1]
        rb1 = s1 / (s1 + 1j * s2)

        qq = np.real((2 * i + 3) * (ra1 + rb1))
        Qext[indices] = Qext[indices] + qq

        if np.all(np.less(np.abs(qq / Qext[indices]), eps)):
            break

        besJ0 = copy.deepcopy(besJ1)
        besJ1 = copy.deepcopy(besJ2)
        besY0 = copy.deepcopy(besY1)
        besY1 = copy.deepcopy(besY2)
        iu0   = iu1
        iu1   = iu2
        ra0[indices]   = copy.deepcopy(ra1)
        rb0[indices]   = copy.deepcopy(rb1)


    Qext = 2.0 / x**2 * Qext
    return Qext

