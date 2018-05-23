import numpy as np
import pickle
import sys

def get_species_from_file(filename):
    species = []
    for line in open(filename):
        species.append(line.split()[0])
    return species

def write_species_file(filename, species):
    with open(filename, "w") as f:
        for s in species:
            f.write(s + "\n")

def read_ggchem_file(filename, species_info_file):
    header = None
    num_atoms = None
    num_molecules = None

    abund_dict = dict()
    abund_arr = []

    for i, line in enumerate(open(filename)):
        elements = line.split()
        if i == 1:
            num_atoms = int(elements[0])
            num_molecules = int(elements[1])
        if i == 2:
            elements[elements.index("COS")] = "OCS"
            elements[elements.index("TIO")] = "TiO"
            elements[elements.index("HCL")] = "HCl"
            elements[elements.index("MGH")] = "MgH"
            elements[elements.index("SIO")] = "SiO"
            elements[elements.index("SIH")] = "SiH"            
            header = np.array(elements)
        if i > 2: break

    data = np.loadtxt(filename, skiprows=3)
    species_to_include = get_species_from_file(species_info_file)
    include = [h in species_to_include for h in header]

    for i,row in enumerate(data):
        total = np.sum(10**row[include])
        grand_total = np.sum(10**row[3 : 4 + num_atoms + num_molecules])
        data[i, include] = 10**row[include]/total
        #print total/grand_total

    N_P = 13
    N_T = data.shape[0]/N_P

    write_species_file("included_species", [h for h in header if h in species_to_include])
    
    for i, h in enumerate(header):
        if h not in species_to_include: continue
        abund_dict[h] = data[:,i].reshape((N_P, N_T))
        abund_dict[h] = np.flip(abund_dict[h], 0)
        abund_dict[h] = np.flip(abund_dict[h], 1)

        if N_T == 55: #T in 50K increments, with 300K min
            include_T = np.arange(N_T) % 2 == 0
            abund_dict[h] = abund_dict[h][:, include_T]

            fake_cols = np.ones((N_P, 2)) * np.nan
            abund_dict[h] = np.append(fake_cols, abund_dict[h], axis=1)
        elif N_T == 28: #T in 50K increments, with 300K min
            fake_cols = np.ones((N_P, 2)) * np.nan
            abund_dict[h] = np.append(fake_cols, abund_dict[h], axis=1)
        elif N_T != 30:
            assert(False)

        abund_arr.append(abund_dict[h])

    return abund_dict, np.array(abund_arr)



'''logZ = sys.argv[1]
filename = "result_{0}/Static_Conc_2D.dat".format(logZ)
species_info_file = sys.argv[2]

abund_dict, abund_arr = read_ggchem_file(filename, species_info_file)

output_filename = "abund_dict_{0}.pkl".format(logZ)
pickle.dump(abund_dict, open(output_filename, "wb"))    
'''
