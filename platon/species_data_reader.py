import numpy as np
import os

def read_species_data(absorption_dir, species_info_file, absorption_file_prefix="absorb_coeffs_"):
    absorption_data = dict()
    mass_data = dict()
    polarizability_data = dict()
    
    with open(species_info_file) as f:
        for line in f:
            if line[0] == '#': continue
            columns = line.split()
            name = columns[0]
            mass = float(columns[1])
            polarizability = float(columns[2])
            absorption_filename = os.path.join(absorption_dir, absorption_file_prefix + name + ".npy")
            if os.path.isfile(absorption_filename):
                absorption_data[name] = np.load(absorption_filename)
            mass_data[name] = mass

            if polarizability != 0:
                polarizability_data[name] = polarizability

    return absorption_data, mass_data, polarizability_data
    

