import os


def read_species_data(absorption_dir, species_info_file, method,
                      include_opacities):
    """Reads the species info table.  Returns (absorption_files, mass_data,
    polarizability_data), where absorption_files maps each included species
    with an opacity file to its path; the caller loads the (large) opacity
    arrays itself, directly into a preallocated stack."""
    if method != "xsec":
        raise NotImplementedError(
            "Correlated-k (ktables) support has been removed; use method='xsec'")
    absorption_file_prefix = "absorb_coeffs_"

    absorption_files = dict()
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
                absorption_files[name] = absorption_filename
            mass_data[name] = mass

            if polarizability != 0:
                polarizability_data[name] = polarizability

    return absorption_files, mass_data, polarizability_data
