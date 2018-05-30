import numpy as np
import matplotlib.pyplot as plt

all_data = np.load("abundances/cond/all_data.npy")
all_CO = np.arange(0.2, 2.2, 0.2)

for i, species in enumerate(np.loadtxt("abundances/cond/included_species", dtype=str)):
    abundances = all_data[7, 12, i, :, 20]
    plt.semilogy(all_CO, abundances, label=species)

plt.xlabel("CO ratio")
plt.ylabel("Abundance")
plt.title("Abundances at 1000 Pa, 1300 K, logZ=0")
plt.legend()
plt.show()
print all_data.shape
