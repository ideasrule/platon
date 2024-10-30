import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.interpolate

pairs = [("H2", "H2"),
         ("H2", "He")]
M_TO_CM = 100
grid_wavelengths = np.exp(np.linspace(np.log(0.2e-6), np.log(30e-6), 5011))
grid_temperatures = np.arange(100, 4100, 100)
output_f = open("collisional_absorption.pkl", "wb")
output_dict = {}

for pair in pairs:
    print("Processing", pair)
    temperatures = []
    wavenums = []
    data = []

    for line in open("{}-{}_2011.cia".format(pair[0], pair[1])):
        elements = line.split()
        if len(elements) != 2:
            temperatures.append(float(elements[4]))
            wavenums.append([])
            data.append([])
        else:
            wavenums[-1].append(float(elements[0]))
            data[-1].append(float(elements[1]))

    temperatures = np.array(temperatures)
    wavenums = np.array(wavenums)[:,::-1]
    print("Found {} temperatures: {}".format(len(temperatures), temperatures))
    print("Found {} wavenums: {}".format(len(wavenums), wavenums))
    data = np.array(data)[:,::-1]

    assert(np.all(wavenums == wavenums[0]))
    wavenums = wavenums[0]

    wavelengths = 1/wavenums/M_TO_CM
    data /= M_TO_CM**5
    
    interp_data = []
    for i in range(len(data)):
        interp_data.append(np.interp(grid_wavelengths, wavelengths, data[i], left=0, right=0))
    interp_data = np.array(interp_data)

    final_data = scipy.interpolate.interp1d(temperatures, interp_data, axis=0, bounds_error=False, fill_value=(interp_data[0], interp_data[-1]))(grid_temperatures)
    output_dict[pair] = final_data

pickle.dump(output_dict, output_f)
