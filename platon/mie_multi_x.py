import numpy as np
import scipy
import scipy.special as spl
import copy
import pdb
import time

def get_iterations_required(refractive_index, x):
    y = np.sqrt(np.real(refractive_index * np.conj(refractive_index))) * x
    num = 1.25 * y + 15.5

    num[y<1.0] = 7.5 * y[y<1.0] + 9.0

    num[(y>100.0) & (y<50000.0)] = 1.0625 * y[(y>100.0) & (y<50000.0)] + 28.5

    num[y>50000.0] = 1.005 * y[y>50000.0] + 50.5

    num = num.astype(int)
    return num

def get_An(max_n, m, xs):
    #Returns An(mx) from n=0 to n = max_n-1 using downward recursion
    An = np.zeros((max_n, len(xs)), dtype=complex)
    An[max_n - 1] = max_n / (m * xs)
    for i in range(max_n - 2, -1, -1):
        An[i] = (i + 1)/(m*xs) - 1.0/((i + 1)/(m*xs) + An[i+1])
    return An
    

def get_Qext(m, xs):
    num_iterations = get_iterations_required(m, xs)
    A = get_An(np.max(num_iterations), m, xs)
    Qext = np.zeros(len(xs))

    #At iteration i, curr_J is the first kind Bessel function of order i+1/2
    curr_Y = np.zeros(len(xs))

    #At iteration i, curr_J is the second kind Bessel function of order i+1/2
    curr_J = np.zeros(len(xs))
    
    prev_Y = -np.sqrt(2/np.pi/xs) * np.cos(xs) #Y_(i-1/2)
    prev_prev_Y = None # Y_(i-3/2) (but i=1 at beginning)
    prev_J = np.sqrt(2/np.pi/xs) * np.sin(xs) #J_(i-1/2)
    
    for i in range(1, np.max(num_iterations)):
        cond = num_iterations > i
        
        if i == 1:
            curr_Y = -np.sqrt(2/np.pi/xs) * (np.cos(xs)/xs + np.sin(xs))
        else:
            curr_Y[cond] = (2*i - 1)/xs[cond] * prev_Y[cond] - prev_prev_Y[cond]

        curr_J[cond] = 2/np.pi/xs[cond]/prev_Y[cond] + curr_Y[cond]/prev_Y[cond]*prev_J[cond]
        
        an_numerator = (A[i][cond]/m + i/xs[cond]) * curr_J[cond] - prev_J[cond]
        an_denominator = an_numerator + 1j*((A[i][cond]/m + i/xs[cond])*curr_Y[cond] - prev_Y[cond])
        an = an_numerator/an_denominator

        bn_numerator = (m*A[i][cond] + i/xs[cond])*curr_J[cond] - prev_J[cond]
        bn_denominator = bn_numerator + 1j*((m*A[i][cond] + i/xs[cond])*curr_Y[cond] - prev_Y[cond])
        bn = bn_numerator/bn_denominator
        Qext[cond] += (2*i + 1) * (an + bn).real

        prev_prev_Y = np.copy(prev_Y)
        prev_J = np.copy(curr_J)
        prev_Y = np.copy(curr_Y)
        
    Qext *= 2/xs**2
    return Qext
        
