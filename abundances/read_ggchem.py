import numpy as np
import pickle
import sys

def get_species_from_file(filename):
    species = []
    for line in open(filename):
        species.append(line.split()[0])
    return species


logZ = sys.argv[1]
filename = "result_{0}/Static_Conc_2D.dat".format(logZ)
species_info_file = sys.argv[2]

N_T = 30
N_P = 13

header = None
num_atoms = None
num_molecules = None

abund_dict = dict()


for i, line in enumerate(open(filename)):
    elements = line.split()
    if i == 1:
        num_atoms = int(elements[0])
        num_molecules = int(elements[1])
    if i == 2: header = elements
    if i > 2: break

data = np.loadtxt(filename, skiprows=3)
species_to_include = get_species_from_file(species_info_file)
include = [h in species_to_include for h in header]

for i,row in enumerate(data):
    total = np.sum(10**row[include])
    data[i, include] = 10**row[include]/total

#print header
for i, h in enumerate(header):
    abund_dict[h] = data[:,i].reshape((N_P, N_T))
    abund_dict[h] = np.flip(abund_dict[h], 0)
    abund_dict[h] = np.flip(abund_dict[h], 1)

output_filename = "abund_dict_{0}.pkl".format(logZ)
pickle.dump(abund_dict, open(output_filename, "wb"))    


