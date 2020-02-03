import numpy as np
import sys
from read_one import read_ggchem_file

def load(dir, species_info_file):    
    metallicities = np.arange(-1, 3.0+0.1, 0.1)
    COs = np.array([0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0])
    all_data = []
    
    for m_index, m in enumerate(metallicities):
        data_for_Z = []
        for co_index, co in enumerate(COs):
            if m < 1e-4 and m > -1e-4: m = 0
            if co == 0.95:
                co_string = "0.95"
            elif co == 1.05:
                co_string = "1.05"
            else:
                co_string = "{:.1f}".format(co)
            filename = "{}/result_{:.1f}_{}/Static_Conc_2D.dat".format(dir, m, co_string)
            print(filename, m)

            abund_dict, abund_arr = read_ggchem_file(filename, species_info_file)
            data_for_Z.append(abund_arr)
            #all_data.append(abund_arr)
        all_data.append(data_for_Z)
            
    all_data = np.array(all_data)

    #Want shape to be: (N_P, N_T, N_species, N_CO, N_metallicites)
    all_data = all_data.transpose([0, 1, 2, 4, 3])
    np.save("all_data.npy", all_data)


load(sys.argv[1], sys.argv[2])
