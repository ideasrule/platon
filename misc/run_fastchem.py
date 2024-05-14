import pyfastchem
import numpy as np
import pdb
import time
import itertools
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os.path
from save_output import saveChemistryOutput, saveMonitorOutput

k_B = 1.38e-16
T_grid = np.arange(100, 3100, 100)
P_grid = 10.0**np.arange(-9, 4)
included_species = ["e-", "H", "H1-", "He", "C", "N", "O", "Na", "Fe", "Ca", "Ti", "K", "Ni", "H2", "N2", "O2", "H1O1",
                    "C1O1", "N1O1", "O1Si1", "O1Ti1", "O1V1", "C1H1N1_1", "C1H4", "C1O2", "H2O1", "H2S1", "H3N1", 
                    "H3P1", "N1O2", "O2S1", "O3", "C2H2"]

def run_for_logZ_CO(params):   
    logZ, CO_ratio = params
    filename = "abundances_{}_{}.npy".format(round(logZ, 5), round(CO_ratio, 5))
    if os.path.isfile(filename):
        return

    print("Processing logZ={}, CO_ratio={}".format(logZ, CO_ratio))
    P, T = np.array([(p, t) for t in T_grid for p in P_grid]).T
    fastchem = pyfastchem.FastChem("/home/stanley/packages/FastChem/input/element_abundances/asplund_2020.dat",
                                   "/home/stanley/packages/FastChem/input/logK/logK.dat",
                                   "/home/stanley/packages/FastChem/input/logK/logK_condensates.dat",
                               1)
    if logZ > 2.9 and CO_ratio < 0.3:
        #Seems to run into numerical issues        
        fastchem.setParameter("minDensityExponentElement", -1920.0)

    #fastchem.setParameter("nbIterationsCond", 30000)
    fastchem.setParameter("nbIterationsChem", 60000)
    
    #fastchem.setParameter("nbIterationsBisection", 30000)
    #fastchem.setParameter("nbIterationsChemCond", 30000)
    #fastchem.setParameter("nbIterationsNelderMead", 30000)
    #fastchem.setParameter("nbIterationsNewton", 30000)

    #fastchem.setParameter("condIterChangeLimit", 2.0)
    #fastchem.setParameter("condSolveFullSystem", np.bool_(True))
    #fastchem.setParameter("condUseFullPivot", np.bool_(True))
    #fastchem.setParameter("condUseSVD", np.bool_(True))
    
    element_abundances = np.array(fastchem.getElementAbundances())
    for i in range(fastchem.getElementNumber()):
        symbol = fastchem.getGasSpeciesSymbol(i)
        if symbol != 'H' and symbol != 'He':
            element_abundances[i] *= 10**logZ
                
    index_C = fastchem.getElementIndex('C')
    index_O = fastchem.getElementIndex('O')
        
    sum_C_O = element_abundances[index_C] + element_abundances[index_O]
    element_abundances[index_O] = sum_C_O / (1 + CO_ratio)
    element_abundances[index_C] = CO_ratio / (1 + CO_ratio) * sum_C_O
        
    fastchem.setElementAbundances(element_abundances)

    #create the input and output structures for FastChem
    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()

    input_data.temperature = T
    input_data.pressure = P
    input_data.equilibrium_condensation = True

    fastchem_flag = fastchem.calcDensities(input_data, output_data)
    print("FastChem reports for {}, {}:".format(logZ, CO_ratio), pyfastchem.FASTCHEM_MSG[fastchem_flag])
    if fastchem_flag != pyfastchem.FASTCHEM_SUCCESS:
        return
    
    number_densities = np.array(output_data.number_densities)
    abundances = []
    for i in range(len(included_species)):
        index = fastchem.getGasSpeciesIndex(included_species[i])
        abundances.append(number_densities[:,index].reshape((len(T_grid), len(P_grid))))

    abundances = np.array(abundances)
    abundances /= abundances.sum(axis=0)
    np.save(filename, abundances)
    return abundances


P, T = np.array([(p, t) for t in T_grid for p in P_grid]).T
reshaped_T = T.reshape((len(T_grid), len(P_grid)))
reshaped_P = P.reshape((len(T_grid), len(P_grid)))

all_logZ = np.linspace(-2, 3, 51)
all_CO_ratios = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0])
abundances = np.zeros((len(all_logZ), len(all_CO_ratios), len(included_species), len(T_grid), len(P_grid)))

logZ_CO_combos = list(itertools.product(all_logZ, all_CO_ratios))
#run_for_logZ_CO(logZ_CO_combos[0])
with Pool() as p:
    p.map(run_for_logZ_CO, logZ_CO_combos)

'''for i, logZ in enumerate(all_logZ):
    for j, CO_ratio in enumerate(all_CO_ratios):
        #print(i,j)
        #if logZ <= 0: continue
        print(logZ, CO_ratio)
        abundances[i,j] = run_for_logZ_CO(logZ, CO_ratio)
        np.save("abundances_{}_{}.npy".format(i,j), abundances[i,j])
        
np.save("with_condensates.npy", abundances)  
#np.save("gas_only_alt.npy", abundances)
'''
