from species_data_reader import read_species_data
import interpolator_3D
import eos_reader
from scipy.interpolate import RectBivariateSpline
from tau_los import get_line_of_sight_tau
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from constants import k_B, amu

class TransitDepthCalculator:
    def __init__(self, planet_radius, star_radius, g, absorption_dir="Absorption", species_masses_file="all_species_masses", lambda_grid_file="wavelengths.npy", P_grid_file="pressures.npy", T_grid_file="temperatures.npy"):
        self.planet_radius = planet_radius
        self.star_radius = star_radius
        self.g = g
        self.absorption_data, self.mass_data = read_species_data(absorption_dir, species_masses_file)

        self.lambda_grid = np.load(lambda_grid_file)
        self.P_grid = np.load(P_grid_file)
        self.T_grid = np.load(T_grid_file)

        self.N_lambda = len(self.lambda_grid)
        self.N_temperatures = len(self.T_grid)
        self.N_pressures = len(self.P_grid)

    def compute_depths(self, P, T, abundances):
        '''P: array of length N_tau
           T: array of length N_tau
           abundances: dictionary mapping species name to (N_T, N_P) array, where N_T is the number of temperature points in the absorption data files, and N_P is the number of pressure points in those files'''
        start = time.time()
        assert(len(P) == len(T))
        
        N_tau = len(P)
        absorption_coeff = np.zeros((self.N_lambda, self.N_pressures, self.N_temperatures))
        mu = np.zeros(N_tau)
        
        for species_name in abundances.keys():
            assert(abundances[species_name].shape == (self.N_pressures, self.N_temperatures))
            if species_name in self.absorption_data: 
                absorption_coeff += self.absorption_data[species_name] * abundances[species_name]
            if species_name in self.mass_data:
                interpolator = RectBivariateSpline(self.P_grid, self.T_grid, abundances[species_name], kx=1, ky=1)
                atm_abundances = interpolator.ev(P, T)
                mu += atm_abundances * self.mass_data[species_name]
            
        absorption_coeff_atm = interpolator_3D.fast_interpolate(absorption_coeff, self.T_grid, self.P_grid, T, P)
        
        dP = P[1:] - P[0:-1]
        dh = dP/P[1:] * k_B * T[1:]/(mu[1:] * amu* self.g)
        dh = np.append(k_B*T[0]/(mu[0] * amu * self.g), dh)
        
        #dz goes from top to bottom of atmosphere
        radius_with_atm = np.sum(dh) + self.planet_radius
        heights = np.append(radius_with_atm, radius_with_atm - np.cumsum(dh))
        tau_los = get_line_of_sight_tau(absorption_coeff_atm, heights)

        absorption_fraction = 1 - np.exp(-tau_los)
        transit_depths = (self.planet_radius/self.star_radius)**2 + 2/self.star_radius**2 * absorption_fraction.dot(heights[1:] * dh)

        end = time.time()
        print "Time taken", end-start
        return transit_depths
        
        
depth_calculator = TransitDepthCalculator(6.4e6, 7e8, 9.8)

index, P, T = np.loadtxt("T_P/t_p_800K.dat", unpack=True, skiprows=1)
abundances = eos_reader.get_abundances("EOS/eos_1Xsolar_cond.dat")

transit_depths = depth_calculator.compute_depths(P, T, abundances)
transit_depths *= 100

ref_wavelengths, ref_depths = np.loadtxt("ref_spectra.dat", unpack=True, skiprows=2)
plt.plot(ref_wavelengths, ref_depths, label="ExoTransmit")
plt.plot(depth_calculator.lambda_grid, transit_depths, label="PyExoTransmit")
plt.legend()
plt.figure()
plt.plot(ref_wavelengths, ref_depths-transit_depths)
plt.show()

