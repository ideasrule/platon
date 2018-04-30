import sys
import numpy as np

def get_abundances(filename):
    '''Reads EOS file in the ExoTransmit format, returning a dictionary mapping species name to an abundance array of dimension'''
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
                temperatures.append(elements[0])
                pressures.append(elements[1])
                compositions.append(elements[2:])

            line_counter += 1

    temperatures = np.array(temperatures)
    pressures = np.array(pressures)
    compositions = np.array(compositions)

    N_temperatures = len(np.unique(temperatures))
    N_pressures = len(np.unique(pressures))
    
    for i in range(len(species)):
        c = compositions[:, i].reshape((N_pressures, N_temperatures))
        #This file has decreasing temperatures and pressures; we want increasing temperatures and pressures
        c = np.flip(c, 0)
        c = np.flip(c, 1)
        abundance_data[species[i]] = c
    return abundance_data
