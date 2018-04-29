import numpy as np
import os
from constants import k_B


def read_species_data(absorption_dir, species_masses_file, absorption_file_prefix="absorb_coeffs_"):
    absorption_data = dict()
    mass_data = dict()
    polarizability_data = dict()
    
    with open(species_masses_file) as f:
        for line in f:
            if line[0] == '#': continue
            columns = line.split()
            species_name = columns[0]
            print "Loading", species_name
            species_mass = float(columns[1])
            polarizability = float(columns[2])
            absorption_filename = os.path.join(absorption_dir, absorption_file_prefix + species_name + ".npy")
            if os.path.isfile(absorption_filename):
                absorption_data[species_name] = np.load(absorption_filename)
            mass_data[species_name] = species_mass

            if polarizability != 0:
                polarizability_data[species_name] = polarizability

    return absorption_data, mass_data, polarizability_data
    

