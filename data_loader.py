import numpy as np

class OpacityData:
    species_name = None
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
        k_B = 1.38e-16
        opacities = np.zeros(self.cross_sections.shape)
        for T_index in range(len(self.temperatures)):
            for P_index in range(len(self.pressures)):
                rho = k_B * self.temperatures[T_index]/self.pressures[P_index]
                opacities[:, P_index, T_index] = self.cross_sections[:, P_index, T_index]/rho
        print opacities
        return opacities



species = ["C2H2", "C2H4"]

for s in species:
    filename = "Opac/opac{0}.dat".format(s)
    data_obj = OpacityData(s, filename)
    #read_opacity_file(filename)
    #data = np.loadtxt(filename)
    #print 
