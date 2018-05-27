from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import nestle

import eos_reader
from transit_depth_calculator import TransitDepthCalculator
from fit_info import FitInfo


class Retriever:
    def ln_prob(self, params, calculator, fit_info, measured_depths,
                measured_errors, plot=False):
        
        if not fit_info.within_limits(params):
            return -np.inf

        params_dict = fit_info.interpret_param_array(params)

        R = params_dict["R"]
        T = params_dict["T"]
        logZ = params_dict["logZ"]
        CO_ratio = params_dict["CO_ratio"]
        scatt_factor = 10.0**params_dict["log_scatt_factor"]
        cloudtop_P = 10.0**params_dict["log_cloudtop_P"]
        error_multiple = params_dict["error_multiple"]

        if not calculator.is_in_bounds(logZ, CO_ratio, T):
            return -np.inf        

        wavelengths, calculated_depths = calculator.compute_depths(
            R, T, logZ, CO_ratio,
            scattering_factor=scatt_factor, cloudtop_pressure=cloudtop_P)
        residuals = calculated_depths - measured_depths
        scaled_errors = error_multiple * measured_errors
        result = -0.5 * np.sum(residuals**2/scaled_errors**2 + np.log(2*np.pi*scaled_errors**2))

        if plot:
            plt.errorbar(1e6*wavelengths, measured_depths, yerr=measured_errors, fmt='.')
            plt.plot(1e6*wavelengths, calculated_depths)
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Transit depth")
            plt.show()
        return result
    
    def run_emcee(self, wavelength_bins, depths, errors, fit_info, nwalkers=50,
                  nsteps=10000, include_condensates=True,
                  output_prefix="output"):        
        initial_positions = fit_info.generate_rand_param_arrays(nwalkers)
        calculator = TransitDepthCalculator(fit_info.get("star_radius"), fit_info.get("g"), include_condensates=include_condensates)
        calculator.change_wavelength_bins(wavelength_bins)        
        
        sampler = emcee.EnsembleSampler(
            nwalkers, fit_info.get_num_fit_params(), self.ln_prob,
            args=(calculator, fit_info, depths, errors))

        for i, result in enumerate(sampler.sample(initial_positions, iterations=nsteps)):
            if (i+1) % 10 == 0:
                print(str(i+1) + "/" + str(nsteps), sampler.lnprobability[0,i], sampler.chain[0,i])
        
        np.save(output_prefix + "_chain.npy", sampler.chain)
        np.save(output_prefix + "_lnprob.npy", sampler.lnprobability)

    def run_multinest(self, wavelength_bins, depths, errors, fit_info, maxiter=None, include_condensates=True):
        calculator = TransitDepthCalculator(fit_info.get("star_radius"), fit_info.get("g"), include_condensates=include_condensates)
        calculator.change_wavelength_bins(wavelength_bins)
        
        def transform_prior(cube):
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                low_guess, high_guess = fit_info.get_guess_bounds(i)
                new_cube[i] = cube[i]*(high_guess - low_guess) + low_guess
            return new_cube

        def multinest_ln_prob(cube):
            return self.ln_prob(cube, calculator, fit_info, depths, errors)

        def callback(callback_info):
            print(callback_info["it"], callback_info["logz"],
                  transform_prior(callback_info["active_u"][0]))
        
        result = nestle.sample(
            multinest_ln_prob, transform_prior, fit_info.get_num_fit_params(),
            callback=callback, method='multi', maxiter=maxiter)
        
        best_params_arr = result.samples[np.argmax(result.logl)]
        best_params_dict = fit_info.interpret_param_array(best_params_arr)
        print("Best params", best_params_dict)
        return result
        
        
        
    def plot_result(self, wavelength_bins, depths, errors, fit_info, parameter_array):
        calculator = TransitDepthCalculator(fit_info.get("star_radius"), fit_info.get("g"))
        calculator.change_wavelength_bins(wavelength_bins)
        self.ln_prob(parameter_array, calculator, fit_info, depths, errors, plot=True)
        


'''def hd209458b_stis():
    #http://iopscience.iop.org/article/10.1086/510111/pdf
    star_radius = 7.826625e8
    jupiter_radius = 7.1492e7
    wave_bins = [[293,347], [348,402], [403,457], [458,512], [512,567], [532,629], [629,726], [727,824], [825,922], [922,1019]]
    wave_bins = 1e-9 * np.array(wave_bins)
    
    planet_radii = [1.3263, 1.3254, 1.32, 1.3179, 1.3177, 1.3246, 1.3176, 1.3158, 1.32, 1.3268]
    radii_errors = [0.0018, 0.0010, 0.0006, 0.0006, 0.0010, 0.0006, 0.0005, 0.0006, 0.0006, 0.0013]
    transit_depths = (np.array(planet_radii)*jupiter_radius/star_radius)**2 + 60e-6
    transit_errors = np.array(radii_errors)/np.array(planet_radii) * 2 * transit_depths
    return wave_bins, transit_depths, transit_errors

def hd209458b_wfc3():
    #https://arxiv.org/pdf/1302.1141.pdf
    wavelengths = 1e-6*np.array([1.119, 1.138, 1.157, 1.175, 1.194, 1.213, 1.232, 1.251, 1.270, 1.288, 1.307, 1.326, 1.345, 1.364, 1.383, 1.401, 1.420, 1.439, 1.458, 1.477, 1.496, 1.515, 1.533, 1.552, 1.571, 1.590, 1.609, 1.628])
    wavelength_bins = [[w-0.0095e-6, w+0.0095e-6] for w in wavelengths]
    depths = 1e-6 * np.array([14512.7, 14546.5, 14566.3, 14523.1, 14528.7, 14549.9, 14571.8, 14538.6, 14522.2, 14538.4, 14535.9, 14604.5, 14685.0, 14779.0, 14752.1, 14788.8, 14705.2, 14701.7, 14677.7, 14695.1, 14722.3, 14641.4, 14676.8, 14666.2, 14642.5, 14594.1, 14530.1, 14642.1])
    errors = 1e-6 * np.array([50.6, 35.5, 35.2, 34.6, 34.1, 33.7, 33.5, 33.6, 33.8, 33.7, 33.4, 33.4, 33.5, 33.9, 34.4, 34.5, 34.7, 35.0, 35.4, 35.9, 36.4, 36.6, 37.1, 37.8, 38.6, 39.2, 39.9, 40.8])
    return np.array(wavelength_bins), depths, errors

def hd209458b_spitzer():
    #https://arxiv.org/pdf/1504.05942.pdf

    wave_bins = []
    depths = []
    errors = []

    wave_bins.append([3.2, 4.0])
    RpRs = np.average([0.12077, 0.1222, 0.11354, 0.11919], weights=1.0/np.array([0.00085, 0.00062, 0.00087, 0.00032]))
    depths.append(RpRs**2)
    errors.append(0.00032/RpRs * 2 * depths[-1])

    wave_bins.append([4.0, 5.0])
    RpRs = np.average([0.12199, 0.12099], weights=1.0/np.array([0.00094, 0.00029]))
    depths.append(RpRs**2)
    errors.append(0.00029/RpRs * 2 * depths[-1])

    wave_bins.append([5.1, 6.3])
    RpRs = np.average([0.12007, 0.11880], weights=1.0/np.array([0.00248, 0.00272]))
    depths.append(RpRs**2)
    errors.append(0.00248/RpRs * 2 * depths[-1])

    wave_bins.append([6.6, 9.0])
    RpRs = np.average([0.12007, 0.11991], weights=1.0/np.array([0.00114, 0.00073]))
    depths.append(RpRs**2)
    errors.append(0.00073/RpRs * 2 * depths[-1])
    
    return 1e-6*np.array(wave_bins), np.array(depths), np.array(errors)

stis_bins, stis_depths, stis_errors = hd209458b_stis()
wfc3_bins, wfc3_depths, wfc3_errors = hd209458b_wfc3()
spitzer_bins, spitzer_depths, spitzer_errors = hd209458b_spitzer()

bins = np.concatenate([stis_bins, wfc3_bins, spitzer_bins])
depths = np.concatenate([stis_depths, wfc3_depths, spitzer_depths])
errors = np.concatenate([stis_errors, wfc3_errors, spitzer_errors])

for i in range(len(bins)):
    print(bins[i][0], bins[i][1], depths[i], errors[i])

#bins = wfc3_bins
#depths = wfc3_depths
#errors = wfc3_errors


#plt.errorbar([(start+end)/2 for (start,end) in bins], depths, yerr=errors, fmt='.')
#plt.show()
        
retriever = Retriever('ggchem', include_condensates=False)

R_guess = 9.7e7
T_guess = 1200
metallicity_guess = 1
scatt_factor_guess = 1
cloudtop_P_guess = 1e6

fit_info = FitInfo({'R': R_guess, 'T': T_guess, 'logZ': np.log10(metallicity_guess), 'CO_ratio': 0.53, 'log_scatt_factor': np.log10(scatt_factor_guess), 'log_cloudtop_P': np.log10(cloudtop_P_guess), 'star_radius': 8.0e8, 'g': 9.311, 'error_multiple': 1})

fit_info.add_fit_param('R', 0.9*R_guess, 1.1*R_guess, 0, np.inf)
fit_info.add_fit_param('T', 0.5*T_guess, 1.5*T_guess, 0, np.inf)
fit_info.add_fit_param('logZ', -1, 3, -1, 3)
fit_info.add_fit_param('CO_ratio', 0.2, 1.5, 0.2, 2.0)
fit_info.add_fit_param('log_cloudtop_P', -1, 4, -np.inf, np.inf)
#fit_info.add_fit_param('log_scatt_factor', 0, 1, 0, 3)
fit_info.add_fit_param('error_multiple', 0.1, 10, 0, np.inf)


result = retriever.run_multinest(bins, depths, errors, fit_info, output_prefix="ggchem_error")

np.save("samples.npy", result.samples)
np.save("weights.npy", result.weights)
np.save("logl.npy", result.logl)
fig = corner.corner(result.samples, weights=result.weights)
fig.savefig("multinest_corner.png")


#retriever.plot_result(bins, depths, errors, fit_info, [1.35868222866*7.1e7, 1108.28033324, np.log10(0.718669990058), np.log10(940.472706829), np.log10(2.87451662752)])
'''
