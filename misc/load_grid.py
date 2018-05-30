import numpy as np
import sys
from read_one import read_ggchem_file

def load(dir, species_info_file):    
    metallicities = np.arange(-1, 3+0.05, 0.05)
    COs = np.arange(0.2, 2.2, 0.2)
    all_data = []
    
    for m_index, m in enumerate(metallicities):
        data_for_Z = []
        for co_index, co in enumerate(COs):
            filename = "{}/result_{:.2f}_{:.1f}/Static_Conc_2D.dat".format(dir, m, co)
            abund_dict, abund_arr = read_ggchem_file(filename, species_info_file)
            print filename
            data_for_Z.append(abund_arr)
            #all_data.append(abund_arr)
        all_data.append(data_for_Z)
            
    all_data = np.array(all_data)

    #Want shape to be: (N_P, N_T, N_species, N_CO, N_metallicites)
    all_data = all_data.transpose([3, 4, 2, 1, 0])
    np.save("all_data.npy", all_data)


load(sys.argv[1], sys.argv[2])
