import numpy as np
import os


def read_species_data(absorption_dir, species_info_file, method,
                      include_opacities, downsample=1):
    if method != "xsec":
        raise NotImplementedError(
            "Correlated-k (ktables) support has been removed; use method='xsec'")
    absorption_file_prefix = "absorb_coeffs_"

    absorption_data = dict()
    mass_data = dict()
    polarizability_data = dict()

    with open(species_info_file) as f:
        for line in f:
            if line[0] == '#':
                continue
            columns = line.split()
            name = columns[0]
            mass = float(columns[1])
            polarizability = float(columns[2])
            absorption_filename = os.path.join(
                absorption_dir, absorption_file_prefix + name + ".npy")
            if os.path.isfile(absorption_filename) and name in include_opacities:
                # float32 is the working precision of the JAX pipeline
                raw = np.load(absorption_filename, mmap_mode="r")
                absorption_data[name] = np.ascontiguousarray(
                    raw[:, :, ::downsample], dtype=np.float32)
            mass_data[name] = mass

            if polarizability != 0:
                polarizability_data[name] = polarizability

    return absorption_data, mass_data, polarizability_data
