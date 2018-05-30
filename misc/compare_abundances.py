from abundance_getter import AbundanceGetter
import eos_reader
import matplotlib.pyplot as plt
import numpy as np


species = "Na"

getter = AbundanceGetter()
ggchem = getter.get(0, 0.6)[species]
exotransmit = eos_reader.get_abundances("EOS/eos_solar_0p6_CtoO_cond.dat")[species]

plt.imshow(ggchem)
plt.title("GGchem")
plt.figure()
plt.title("ExoTransmit")
plt.imshow(exotransmit)

plt.figure()
ratio = exotransmit/ggchem
ratio[ratio > 10] = 10
plt.imshow(ratio, extent=(100, 3000, -4, 8), aspect=100)
plt.xlabel("Temperature (K)")
plt.ylabel("log P(Pa)")
plt.title("exotransmit/ggchem abundances for " + species)
plt.colorbar()
plt.show()
