import sys
import numpy as np

ref_abund = sys.argv[1]
logZ = float(sys.argv[2])
if len(sys.argv) == 4:
    CO = float(sys.argv[3])
else:
    CO = None

if CO == 1:
    CO -= 0.001 #Hack to get around ggchem numerical problems at exactly 1
    
names = []
dexes = []

for line in open(ref_abund):
    elements = line.split()
    name = elements[0]
    dex = float(elements[1])
    if name != 'H' and name != 'He':
        dex += logZ
    names.append(name)
    dexes.append(dex)

if CO is not None:    
    C_index = names.index("C")
    O_index = names.index("O")
    dexes[C_index] = np.log10(CO) + dexes[O_index]

for i in range(len(names)):
    print names[i], dexes[i]
