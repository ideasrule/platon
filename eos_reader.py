import sys
import numpy as np

def get_abundances(filename):
    line_counter = 0

    species = None
    temperatures = []
    pressures = []
    compositions = []
    abundance_data = dict()

    with open(filename) as f:
        for line in f:
            elements = line.split()
            if line_counter == 0:
                assert(elements[0] == 'T')
                assert(elements[1] == 'P')
                species = elements[2:]
            elif len(elements) > 1:
                elements = np.array([float(e) for e in elements])
                pressures.append(elements[0])
                temperatures.append(elements[1])
                #compositions.append(elements[2:]/np.sum(elements[2:]))
                compositions.append(elements[2:])
                #assert(np.abs(np.sum(compositions[-1]) - 1) < 1e-10)
            line_counter += 1

    temperatures = np.array(temperatures)
    pressures = np.array(pressures)
    compositions = np.array(compositions)

    N_temperatures = len(np.unique(temperatures))
    N_pressures = len(np.unique(pressures))
    
    for i in range(len(species)):
        c = compositions[:, i].reshape((N_temperatures, N_pressures))
        #This file has decreasing temperatures and pressures; we want increasing temperatures and pressures
        c = np.flip(c, 0)
        c = np.flip(c, 1)
        abundance_data[species[i]] = c
    return abundance_data
