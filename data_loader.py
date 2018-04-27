import numpy as np
import os

class OpacityData:
    species_name = None
    species_mass = None
    filename = None
    wavelengths = None
    pressures = None
    cross_sections = []
    opacities = []
    
    def __init__(self, species_name, filename):
        self.species_name = species_name
        self.filename = filename
        self.wavelengths, self.temperatures, self.pressures, self.cross_sections = self.read_opacity_file(filename)
        self.opacities = self.get_opacity()

    def read_opacity_file(self, filename):
        line_counter = 0
        temperatures = None

        pressures = None        
        wavelengths = []
        cross_sections = []

        with open(filename) as f:
            curr_wavelength = None

            for line in f:
                elements = [float(e) for e in line.split()]
                if line_counter == 0: temperatures = np.array(elements)
                elif line_counter == 1: pressures = np.array(elements)
                else:
                    if len(elements) == 1:
                        curr_wavelength = elements[0]
                        wavelengths.append(curr_wavelength)
                        cross_sections.append([])
                    else:
                        log_P = elements[0]
                        cross_sections[-1].append(np.array(elements[1:]))
                line_counter += 1

        return np.array(wavelengths), np.array(temperatures), np.array(pressures), np.array(cross_sections)

    def get_opacity(self):
        k_B = 1.38e-23
        opacities = np.zeros(self.cross_sections.shape)
        for T_index in range(len(self.temperatures)):
            for P_index in range(len(self.pressures)):
                rho = k_B * self.temperatures[T_index]/self.pressures[P_index]
                opacities[:, P_index, T_index] = self.cross_sections[:, P_index, T_index]/rho                
        return opacities


def read_species_data(opacity_dir, species_masses_file):
    opacities_data = dict()
    mass_data = dict()
    
    with open(species_masses_file) as f:
        for line in f:
            if line[0] == '#': continue
            columns = line.split()
            species_name = columns[0]
            print species_name
            species_mass = float(columns[1])
            opacity_filename = os.path.join(opacity_dir, "opac{0}.dat".format(species_name))
            if os.path.isfile(opacity_filename):
                opacities_data[species_name] = OpacityData(species_name, opacity_filename)
            mass_data[species_name] = species_mass

    return opacities_data, mass_data
    

