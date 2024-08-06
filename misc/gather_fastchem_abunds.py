import numpy as np
import matplotlib.pyplot as plt
import os

all_logZ = np.linspace(-2, 3, 51)
all_CO_ratios = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95,
                          0.975,
                          1.0, 1.025, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0])
#all_CO_ratios = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0])
abundances = []

for z, logZ in enumerate(all_logZ):
    abundances.append([])
    for c, CO_ratio in enumerate(all_CO_ratios):
        filename = "abundances_{}_{}.npy".format(round(logZ, 5), round(CO_ratio, 5))
        if not os.path.isfile(filename):
            print("Can't find file; replacing with nan's", filename)
            curr_abunds = abundances[0][0] * np.nan
        else:
            curr_abunds = np.load(filename)

        abundances[-1].append(curr_abunds)

abundances = np.array(abundances)
print(abundances.shape)
#np.save("with_condensation.npy", abundances)
np.save("gas_only.npy", abundances)
