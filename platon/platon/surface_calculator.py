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
    def __init__(self,T_star, R_star, a, R_planet, surface_type, use_HES2012 = True, use_new = False, surface_texture = None,
                 stellar_blackbody = True, path_to_own_stellar_spectrum = None, factor = None,
                 custom_GeoA_path = None, custom_cef_path = None, use_custom_rh = False, custom_rh_wavelengths = None,
                 custom_rh = None):
        self.irrad = 0
        self.stellar_blackbody = stellar_blackbody
        self.T_star = T_star * u.K
        self.R_star = R_star * u.m # * R_sun #ExoMAST
        self.a = a * u.m # AU; ExoMAST
        self.R_planet = R_planet * u.m# * R_jup #m
        self.A_bond_conversion = (3/2)

        #define whether to use surfaces from Hu et al. 2012 (HES2012), the new spectral library, or custom
        self.use_HES2012 = use_HES2012
        self.use_new = use_new
        self.use_custom_rh = use_custom_rh

        #this is data corresponding to the HES2012 surfaces
        self.geoa = pd.read_csv('../data/HES2012/new_GeoA.csv', sep = '\t')
        # self.geoa = pd.read_csv('/Users/kimparagas/Desktop/rh_i0_renyus_surfaces.csv')
        self.wavelengths = self.geoa['Wavelength'].to_numpy()
        
        diff = np.diff(self.wavelengths)
        bins = self.wavelengths[1:] - diff/2
        bins_tg = np.concatenate(([self.wavelengths[0] + (bins[0] - self.wavelengths[1])], bins, [self.wavelengths[-1] - (bins[-1] - self.wavelengths[-1])]))
        self.bins = np.zeros([(len(bins_tg) -1), 2])
        for i in np.arange(len(bins_tg) -1):
            self.bins[i] = [bins_tg[i], bins_tg[i+1]]
            
        if self.use_HES2012 == True and self.use_new == True:
            print("Need to decide between the surfaces in Hu et al. 2012, or the surfaces in Paragas et al. 2024.")
            sys.exit()

        self.surface_type = surface_type #define surface type in either case
        self.surface_texture = surface_texture #define surface texture if using Paragas et al. 2024 spectral library
        if self.use_HES2012 == True and self.use_new == False: #so using HES 2012
            # ssa = ascii.read('/Users/kimparagas/Desktop/research/main/jwst_project/code/from_renyu/Crust_SSA.dat',
            #                 delimiter = '\t', header_start = 2, data_start = 3)
            hemi_refls = pd.read_csv('../data/HES2012/rh_of_renyus_surfaces.csv') #load in hemispherical reflectances (rh) of HES 2012 surfaces
            self.rh_wavelengths = hemi_refls['Wavelength'].to_numpy()
            self.rh = hemi_refls[f'{self.surface_type}'].to_numpy()
            self.rh_og = hemi_refls[f'{self.surface_type}'].to_numpy()

            gamma = (1 - self.rh) / (1 + (2 * 1 * self.rh))
            self.ssa = 1 - gamma**2
            r0 = (1 - gamma) / (1 + gamma)
            self.hemispherical_emissivity = (1 - r0) * (1 + (r0 / 6))
            self.hemispherical_emissivity_og = (1 - r0) * (1 + (r0 / 6))
            self.directional_emissivity = 1 - self.rh
            self.directional_emissivity_og = 1 - self.rh_og

            self.surface_geoa_og = self.geoa[self.surface_type]
            self.surface_geoa = self.geoa[self.surface_type]
            self.surface_emi = 1 - self.surface_geoa
        
            self.crust_emission_flux = ascii.read('../data/HES2012/Crust_EmissionFlux.dat', delimiter = '\t')
        
        if self.use_new == True:
            hemi_refls = pd.read_csv(f'../data/Paragas_spectral_library/scaled_hemi_refls.csv', index_col = 0) #load in rh for Paragas et al spectral library
            self.rh_wavelengths = hemi_refls.index.to_numpy() * 1e-6

            if self.surface_type is None and self.surface_texture is None:
                print('Must define a surface type and texture (slab, crushed, or powder) in the Paragas et al. 2024 spectral library.')
                sys.exit()
            if self.surface_type is None:
                print('Must define a surface type in the Paragas et al. 2024 spectral library.')
                sys.exit()
            if self.surface_texture is None:
                print('Must define a surface texture (slab, crushed, powder) in the Paragas et al. 2024 spectral library.')
                sys.exit()

            self.rh = hemi_refls[f'{self.surface_type}_{self.surface_texture}'].to_numpy()
            self.rh_og = hemi_refls[f'{self.surface_type}_{self.surface_texture}'].to_numpy()
            # self.rh = np.interp(self.wavelengths, self.rh_wavelengths, self.rh)

            gamma = (1 - self.rh) / (1 + (2 * 1 * self.rh))
            self.ssa = 1 - gamma**2
            r0 = (1 - gamma) / (1 + gamma)
            self.hemispherical_emissivity = (1 - r0) * (1 + (r0 / 6))
            self.hemispherical_emissivity_og = (1 - r0) * (1 + (r0 / 6))
            self.directional_emissivity = 1 - self.rh
            self.directional_emissivity_og = 1 - self.rh_og

            self.crust_emission_flux = ascii.read(f'../data/Paragas_spectral_library/crustemissionfluxes/{self.surface_type}_crust_emission_flux.csv',
                                                names = ['Temperature [K]', 'slab', 'crushed', 'powder'])

        if self.use_custom_rh == True: #this needs to be checked + redone if we decide to implement this feature
            self.custom_rh_wavelengths = custom_rh_wavelengths
            self.custom_rh = custom_rh
            if self.custom_rh_wavelengths is None:
                print('Must define an array of values for the wavelengths in meters to use for custom hemispherical reflectances.')
                sys.exit()
            if self.custom_rh is None:
                print('Must define an array of values for the custom hemispherical reflectances.')
                sys.exit()
            if len(self.custom_rh_wavelengths) != len(self.custom_rh):
                print('Custom hemispherical reflectance and its corresponding wavelengths do not match in length.')
                sys.exit()
            self.rh_wavelengths = self.custom_rh_wavelengths
            # self.custom_rh = np.interp(self.wavelengths, self.custom_rh_wavelengths, self.custom_rh)
            self.rh = self.custom_rh
            self.rh_og = custom_rh
            gamma = (1 - self.custom_rh) / (1 + (2 * 1 * self.custom_rh))
            self.ssa = 1 - gamma**2
            r0 = (1 - gamma) / (1 + gamma)
            self.hemispherical_emissivity = (1 - r0) * (1 + (r0 / 6))
            self.hemispherical_emissivity_og = (1 - r0) * (1 + (r0 / 6))
            self.directional_emissivity = 1 - self.rh
            self.directional_emissivity_og = 1 - self.rh_og
            temps = np.exp(np.linspace(np.log(100),np.log(1500),10000))
            flux = []
            for T in temps:
                pl = (2 * const.h.si.value * const.c.si.value**2) / self.wavelengths**5 / (np.exp(const.h.si.value * const.c.si.value / self.wavelengths / const.k_B.si.value / T) - 1)
                flux += [np.pi * np.trapz(y = self.hemispherical_emissivity * pl, x = self.wavelengths)] 
            flux = (np.array(flux) * u.J / u.m**2 / u.s).to(u.W/u.m**2).value
            self.crust_emission_flux = pd.DataFrame(columns = ['Temperature [K]', f'{self.surface_texture}'])
            self.crust_emission_flux['Temperature [K]'] = temps
            self.crust_emission_flux[f'flux'] = flux
            # self.geoa = pd.DataFrame(columns = ['Wavelength', f'{self.surface_type}'])
            # self.geoa['Wavelength'] = self.custom_rh_wavelengths
            # self.geoa[f'{self.surface_type}'] = self.custom_rh
            # self.surface_geoa_og = self.custom_rh
            # self.surface_geoa = self.custom_rh
            # self.surface_emi = 1 - self.surface_geoa


        # Teq = (1/4)**(1/4) * self.T_star * np.sqrt(self.R_star / self.a)
        # Teq = Teq.si.value
        # f_poly_coeffs = pd.read_csv('/Users/kimparagas/Desktop/research/main/f_results/f_poly_coeffs.csv', sep = '\t', index_col = 0)
        # poly_models = []
        # if self.surface_type == 'orlando_gold_granite': coeffs = f_poly_coeffs.loc['Granitoid']
        # else: coeffs = f_poly_coeffs.loc[self.surface_type]
        # for i, (c, name) in enumerate(zip(coeffs, coeffs.keys())):
        #     if np.isnan(c) == True:
        #         coeffs[name] = 0
        # if self.surface_type == 'orlando_gold_granite': coeffs = f_poly_coeffs.loc['Granitoid'].to_numpy()
        # else: coeffs = f_poly_coeffs.loc[self.surface_type].to_numpy()
        # coeffs = np.flip(coeffs)

        # poly_model = np.poly1d(coeffs)
        
        # if Teq <= 1064 and Teq >= 300:
        #     factor = poly_model(Teq)
        
        # else:
        #     if Teq < 300:
        #         print('Full redistribution equilibrium temperature is colder than 300 K (the lower boundary of our grid), but will use factors corresponding to 300 K.')
        #         Teq = 300
        #         factor = poly_model(Teq)
                
        #     if Teq > 1064:
        #         print(f'WARNING: Full redistribution equilibrium temperature {Teq:.2f} K is hotter than 1064 K.\nDayside may be (partially) molten in the corresponding 2D models.\nWill use factors corresponding to 1064 K (ensuring a non-molten dayside).')
        #         Teq = 1064
        #         factor = poly_model(Teq)
        self.factors = {'Metal-rich': 0.6052, 'Ultramafic': 0.5532, 'Feldspathic': 0.5414,
        'Basaltic': 0.6004, 'Granitoid': 0.5290, 'Clay': 0.5, 'Ice-rich silicate': 0.5,
        'Fe-oxidized': 0.5978}
        if factor is None: 
            if self.use_new == True: 
                new_factors = pd.read_csv('../data/Paragas_spectral_library/f_relation_new_samples.csv') #factors corresponding to Teq = 1000 K
                self.factor = new_factors[f"{self.surface_type}_{self.surface_texture}"].values[0]
            if self.use_HES2012 == True: 
                self.factor = self.factors[self.surface_type] #factors corresponding to Teq = 1000 K
            if self.use_custom_rh == True: 
                self.factor = 0.5#placeholder value for this option, if we decide to implement this, we'll need to
                                 #code up a method for estimating the average hemispherical reflectance and using that with the linear fit to the other f values
                                 #to estimate its f factor
        if factor is not None:
            self.factor = factor #self defined factor
        
        columns = ['Wavelength', 'Flux', 'Depth']
        self.surface_model = pd.DataFrame(columns = columns)
        self.temperature = 0
        
        self.path_to_own_stellar_spectrum = path_to_own_stellar_spectrum
        
        atm = AtmosphereSolver(include_condensation=True, method="xsec")
        self.lambda_grid = atm.lambda_grid
        if self.path_to_own_stellar_spectrum is None: #PLATON stellar spectrum
            stellar_photon_flux, _ = atm.get_stellar_spectrum(atm.lambda_grid, T_star = self.T_star.si.value, T_spot = None,
                                                              spot_cov_frac = None, blackbody = self.stellar_blackbody)
            stellar_photon_flux = stellar_photon_flux * u.photon / u.s / u.m**2
            wl = atm.lambda_grid * u.m
            self.wl = wl.si.value
            stellar_flux = ((stellar_photon_flux / (u.photon * np.gradient(wl))) * ((const.c * const.h) / (wl))).to(u.W/u.m**2/u.um)
            self.stellar_flux = stellar_flux.to(u.W/u.m**3).value
            
        
        ########################### USER-DEFINED STELLAR SPECTRA ###########################
        #################################################################################################
        
        if self.path_to_own_stellar_spectrum is not None:
            spectrum = pd.read_csv(self.path_to_own_stellar_spectrum) #assumed that the stellar flux is in photons per second
                                                                      #per meter squared per wavelength bin
            wl = spectrum['wavelength'].to_numpy() * u.m
            sf = spectrum['stellar flux'].to_numpy()

            stellar_photon_flux = sf * u.photon / u.s / u.m**2
            stellar_flux = ((stellar_photon_flux / (u.photon * np.gradient(wl))) * ((const.c * const.h) / (wl))).to(u.W/u.m**2/u.um)
            
            self.stellar_flux = stellar_flux.to(u.W/u.m**3).value
            self.wl = wl.si.value
        
    def calc_new_albedo_and_emi(self, plot = False): 
        #basically interpolates the data that we definied in the init portion to the PLATON wavelength space
        if self.use_HES2012:
            interp_albedo = np.interp(self.wavelengths, self.geoa['Wavelength'], self.surface_geoa_og)
            self.surface_geoa_old = self.surface_geoa
            self.surface_geoa = interp_albedo
            
            self.surface_emi_old = self.surface_emi
            self.surface_emi = 1 - self.surface_geoa

            self.rh = np.interp(self.wavelengths, self.rh_wavelengths, self.rh_og)

            self.directional_emissivity = np.interp(self.wavelengths, self.rh_wavelengths, self.directional_emissivity_og)
            self.hemispherical_emissivity = np.interp(self.wavelengths, self.rh_wavelengths, self.hemispherical_emissivity_og)

            if plot:
                plt.plot(self.geoa['Wavelength'] * 1e6, self.surface_geoa_old, color = 'red', label = 'old')
                plt.plot(self.wavelengths * 1e6, self.surface_geoa, 'k--', label = 'new')
                plt.title(f'{self.surface_type} - changed')
                plt.xlabel('wavelength')
                plt.ylabel('albedo') 
                plt.legend()   
                plt.show()
                plt.close()  
        if self.use_new == True or self.use_custom_rh == True:
            self.rh = np.interp(self.wavelengths, self.rh_wavelengths, self.rh_og)

            self.directional_emissivity = np.interp(self.wavelengths, self.rh_wavelengths, self.directional_emissivity_og)
            self.hemispherical_emissivity = np.interp(self.wavelengths, self.rh_wavelengths, self.hemispherical_emissivity_og)

            
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
        emitted_flux_p = np.pi * self.directional_emissivity * self.planck_function(x, temp)
        refl_flux_p = stellar_flux * ((self.R_star / self.a)**2) * self.rh

        total_flux = (emitted_flux_p + refl_flux_p) 
        
        depths = ((total_flux) / (stellar_flux)) * (self.R_planet / self.R_star)**2

        refl_depths = ((refl_flux_p) / (stellar_flux)) * (self.R_planet / self.R_star)**2
        emitted_depths = ((emitted_flux_p) / (stellar_flux)) * (self.R_planet / self.R_star)**2
        return total_flux, depths, refl_depths, emitted_depths
    
    def calc_surface_fluxes(self, skip_temp_calc = True):
        if skip_temp_calc == False: #don't need to go the temperature calculation after the first iteration
            self.mask = np.where(((self.wavelengths >= self.wl[0]) & (self.wavelengths <= self.wl[-1])))[0]
            self.wavelengths = self.wavelengths[self.mask]
            self.calc_new_albedo_and_emi()
            
        stellar_flux = np.interp(self.wavelengths, self.wl, self.stellar_flux)
                
        def calc_temps(x, redist_factor):
            irrad = redist_factor * (np.trapz(y = (stellar_flux) * ((self.R_star / self.a)**2) * (1-self.rh), x = x))
            self.irrad = irrad
            # temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surfaces[i]].value - irrad).argmin()] #jwst_project
            if self.use_new:
                temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surface_texture].data - irrad).argmin()] #clr
            if self.use_custom_rh:
                temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux['flux'].data - irrad.values).argmin()]
            if self.use_HES2012:
                temp = self.crust_emission_flux['Temperature [K]'][np.abs(self.crust_emission_flux[self.surface_type].data - irrad).argmin()]
            return temp
        
        if skip_temp_calc != True:
            self.temperature = calc_temps(self.wavelengths, self.factor)
            self.temperature_og = self.temperature

        total_fluxes, model_depths, reflected_depths, emitted_depths = self.calc_fluxes_and_depths(self.wavelengths,
                                                                                                   self.temperature, stellar_flux)
            
        if skip_temp_calc == False:
            self.surface_model['Flux'] = total_fluxes
            self.surface_model['Depth'] = model_depths
            self.surface_model['Reflected Depth'] = reflected_depths
            self.surface_model['Emitted Depth'] = emitted_depths
        
        if skip_temp_calc:
            self.surface_model = pd.DataFrame(columns = ['Wavelength', 'Flux', 'Depth'])
            self.surface_model['Flux'] = total_fluxes
            self.surface_model["Depth"] = model_depths
            self.surface_model['Reflected Depth'] = reflected_depths
            self.surface_model['Emitted Depth'] = emitted_depths
        
        self.surface_model['Wavelength'] = self.wavelengths
                
    def read_in_temp(self, temp):
        #use user-defined temperature instead
        self.temperature = temp
        self.calc_surface_fluxes()

    def change_spectra(self, wavelengths):
        self.wavelengths = wavelengths
        self.calc_new_albedo_and_emi()
        self.surface_model = pd.DataFrame(columns = ['Wavelength', 'Flux', 'Depth'])
        self.surface_model['Wavelength'] = self.wavelengths
        self.calc_surface_fluxes()
        
    def calc_initial_spectra(self, skip_temp_calc = False):
        self.calc_surface_fluxes(skip_temp_calc = skip_temp_calc)
        