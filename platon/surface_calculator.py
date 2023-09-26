from unittest import skip
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import time

from .constants import h, c, k_B, R_jup, M_jup, R_sun, AU
from ._atmosphere_solver import AtmosphereSolver

from astropy.io import ascii
import pandas as pd
import copy 
from scipy.stats import binned_statistic
import astropy.units as u
import astropy.constants as const
from scipy import interpolate
import sys

class SurfaceCalculator:
    def __init__(self,T_star, R_star, a, R_planet, surface_type, stellar_blackbody = True, path_to_own_stellar_spectrum = None):
        self.stellar_blackbody = stellar_blackbody
        self.T_star = T_star * u.K
        self.R_star = R_star * u.m # 
        self.a = a * u.m # AU; 
        self.R_planet = R_planet * u.m# 
        self.A_bond_conversion = (3/2)
        
        self.geoa = pd.read_csv('../data_kim/new_GeoA.csv', sep = '\t')
        
        self.wavelengths = self.geoa['Wavelength'].to_numpy()
        
        diff = np.diff(self.wavelengths)
        bins = self.wavelengths[1:] - diff/2
        bins_tg = np.concatenate(([self.wavelengths[0] + (bins[0] - self.wavelengths[1])], bins, [self.wavelengths[-1] - (bins[-1] - self.wavelengths[-1])]))
        self.bins = np.zeros([(len(bins_tg) -1), 2])
        for i in np.arange(len(bins_tg) -1):
            self.bins[i] = [bins_tg[i], bins_tg[i+1]]
            
        self.surface_type = surface_type
        self.surface_geoa_og = self.geoa[self.surface_type]
        self.surface_geoa = self.geoa[self.surface_type]
        self.surface_emi = 1 - self.surface_geoa
        
        self.crust_emission_flux = ascii.read('../data_kim/Crust_EmissionFlux.dat', delimiter = '\t')
        
        Teq = (1/4)**(1/4) * self.T_star * np.sqrt(self.R_star / self.a)
        Teq = Teq.si.value
        f_poly_coeffs = pd.read_csv('../data_kim/f_poly_coeffs.csv', sep = '\t', index_col = 0)
        poly_models = []
        coeffs = f_poly_coeffs.loc[self.surface_type]
        for i, (c, name) in enumerate(zip(coeffs, coeffs.keys())):
            if np.isnan(c) == True:
                coeffs[name] = 0
        coeffs = f_poly_coeffs.loc[self.surface_type].to_numpy()
        coeffs = np.flip(coeffs)

        poly_model = np.poly1d(coeffs)
        
        if Teq <= 1064 and Teq >= 300:
            factor = poly_model(Teq)
        
        else:
            if Teq < 300:
                print('Full redistribution equilibrium temperature is colder than 300 K (the lower boundary of our grid), but will use factors corresponding to 300 K.')
                Teq = 300
                factor = poly_model(Teq)
                
            if Teq > 1064:
                print(f'WARNING: Full redistribution equilibrium temperature {Teq:.2f} K is hotter than 1064 K.\nDayside may be (partially) molten in the corresponding 2D models.\nWill use factors corresponding to 1064 K (ensuring a non-molten dayside).')
                Teq = 1064
                factor = poly_model(Teq)
        self.factor = np.array(factor)
        
        columns = ['Wavelength', 'Flux', 'Depth']
        self.surface_model = pd.DataFrame(columns = columns)
        self.temperature = 0
        
        self.path_to_own_stellar_spectrum = path_to_own_stellar_spectrum
        
    def calc_new_albedo_and_emi(self, plot = False): 
        interp_albedo = np.interp(self.wavelengths, self.geoa['Wavelength'], self.surface_geoa_og)
        self.surface_geoa_old = self.surface_geoa
        self.surface_geoa = interp_albedo
        
        self.surface_emi_old = self.surface_emi
        self.surface_emi = 1 - self.surface_geoa

        if plot:
            plt.plot(self.geoa['Wavelength'] * 1e6, self.surface_geoa_old, color = 'red', label = 'old')
            plt.plot(self.wavelengths * 1e6, self.surface_geoa, 'k--', label = 'new')
            plt.title(f'{self.surface_type} - changed')
            plt.xlabel('wavelength')
            plt.ylabel('albedo') 
            plt.legend()   
            plt.show()
            plt.close()  
            
    def planck_function_sum(self, bins, T):
        """
        wl: array of wavelengths in meters
        T: effective temp of object in Kelvin 
        """
        flux = np.array([])
        for b in bins: 
            l = (2 * h * c**2) / (b**5)
            r = 1 / np.expm1((h * c) / (b * k_B * T))
            integral = np.sum(r*l) / 2
            flux = np.append(flux, integral)
        return flux
    
    def planck_function(self, wl, T):
        """
        wl: array of wavelengths in meters
        T: effective temp of object in Kelvin 
        """
        flux = np.array([])

        l = (2 * h * c**2) / (wl**5)
        r = 1 / np.expm1((h * c) / (wl * k_B * T))
        flux = np.append(flux, r*l)
        
        return flux 
    
    def calc_fluxes_and_depths(self, x, temp, stellar_flux):
        emitted_flux_p = np.pi * self.surface_emi * self.planck_function(x, temp)
            
        refl_flux_p = stellar_flux * ((self.R_star / self.a)**2) * self.surface_geoa


        total_flux = (emitted_flux_p + refl_flux_p) 
        
        depths = ((total_flux) / (stellar_flux)) * (self.R_planet / self.R_star)**2
        
        return total_flux, depths    
    
    def calc_surface_fluxes(self, skip_temp_calc = True):
        if self.path_to_own_stellar_spectrum is None:
            atm = AtmosphereSolver(include_condensation=True, method="xsec")
            
            stellar_photon_flux, _ = atm.get_stellar_spectrum(atm.lambda_grid, T_star = self.T_star.si.value, T_spot = None, spot_cov_frac = None, blackbody = self.stellar_blackbody) * u.photon / u.s / u.m**2
            wavelengths = atm.lambda_grid * u.m
            stellar_flux = ((stellar_photon_flux / (u.photon * np.gradient(wavelengths))) * ((const.c * const.h) / (wavelengths))).to(u.W/u.m**2/u.um)
            stellar_flux = stellar_flux.to(u.W/u.m**3).value
            
            stellar_flux = np.interp(self.wavelengths, wavelengths.si.value, stellar_flux)
            
        
        ########################### SELF-DEFINED STELLAR SPECTRA IN PHOTONS/S/M**2 ###########################
        #################################################################################################
        
        if self.path_to_own_stellar_spectrum is not None:
            spectrum = pd.read_csv(self.path_to_own_stellar_spectrum, sep = '\t') #assumed that the stellar flux is in photons per second per meter squared
            wl = spectrum['wavelength'].to_numpy() * u.m
            stellar_photon_flux = spectrum['stellar flux'].to_numpy() * u.photon / u.s / u.m**2
            
            stellar_flux = ((stellar_photon_flux / (u.photon * np.gradient(wl))) * ((const.c * const.h) / (wl))).to(u.W/u.m**2/u.um)
            stellar_flux = stellar_flux.to(u.W/u.m**3).value
            wl = wl.si.value
            
            if skip_temp_calc == False:
                self.mask = np.where(((self.wavelengths >= wl[0]) & (self.wavelengths <= wl[-1])))[0]
                self.wavelengths = self.wavelengths[self.mask]
                self.calc_new_albedo_and_emi()
            
            stellar_flux = np.interp(self.wavelengths, wl, stellar_flux)

        def calc_temps(x, redist_factor):
            Ag = np.mean(self.surface_geoa_og)
            As = (3/2) * Ag
            irrad = redist_factor * (np.trapz(y = (stellar_flux) * ((self.R_star / self.a)**2) * (1-As), x = x)) #do not need pi in this if the stellar spectrum is used vs the planck function 
            # temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surfaces[i]].value - irrad).argmin()] #jwst_project
            temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surface_type].data - irrad).argmin()] #clr
            print(f'{self.surface_type}: {irrad:.2f} W/m², {temp:.1f} K')
            return temp
        
        if skip_temp_calc != True:
            self.temperature = calc_temps(self.wavelengths, self.factor)
            self.temperature_og = self.temperature

        total_fluxes, model_depths = self.calc_fluxes_and_depths(self.wavelengths, self.temperature, stellar_flux)
            
        
        if skip_temp_calc == False:
            self.surface_model['Flux'] = total_fluxes
            self.surface_model['Depth'] = model_depths
        
        if skip_temp_calc:
            self.surface_model = pd.DataFrame(columns = ['Wavelength', 'Flux', 'Depth'])
            self.surface_model['Flux'] = total_fluxes
            self.surface_model["Depth"] = model_depths
        
        self.surface_model['Wavelength'] = self.wavelengths
       
                
    def read_in_temp(self, temp):
        self.temperature = temp

    def calc_initial_spectra(self, skip_temp_calc = False):
        self.calc_surface_fluxes(skip_temp_calc = skip_temp_calc, plot_stellar_spectum = False, plot_surface_spectra = False)
        
    def change_spectra(self, wavelengths):
        self.wavelengths = wavelengths
        self.calc_new_albedo_and_emi()
        self.surface_model = pd.DataFrame(columns = ['Wavelength', 'Flux', 'Depth'])
        self.surface_model['Wavelength'] = self.wavelengths
        self.calc_surface_fluxes()
        