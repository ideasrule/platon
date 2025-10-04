import matplotlib.pyplot as plt
import numpy as np
import corner
from .constants import METRES_TO_UM, BAR_TO_PASCALS, R_jup
from .retrieval_result import RetrievalResult
from . TP_profile import Profile
from . import _cupy_numpy as xp
import dynesty

default_style = ['default',
    {   'font.size': 12,
        'xtick.top': True,
        'xtick.direction': 'out',
        'ytick.right': True,
        'ytick.direction': 'out',
        }]
plt.style.use(default_style)

class Plotter():
    def __init__(self):
        pass


    def plot_retrieval_TP_profiles(self, retrieval_result, plot_samples=False, plot_1sigma_bounds=True, num_samples=100, prefix=None):
        """
        Input a RetrievalResult object to make a plot of the best fit temperature profile 
        and 1 sigma bounds for the profile and/or plot samples of the temperature profile.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        if retrieval_result.retrieval_type == "dynesty":
            equal_samples = dynesty.utils.resample_equal(retrieval_result.samples, retrieval_result.weights)
            np.random.shuffle(equal_samples)
        elif retrieval_result.retrieval_type == "pymultinest":
            equal_samples = retrieval_result.equal_samples
        elif retrieval_result.retrieval_type == "emcee":
            equal_samples = np.copy(retrieval_result.flatchain)
        else:
            assert(False)

        indices = np.random.choice(len(equal_samples), num_samples)
        profile_type = retrieval_result.fit_info.all_params['profile_type'].best_guess
        t_p_profile = Profile()
        profile_pressures = xp.cpu(t_p_profile.pressures)

        temperature_arr = []
        for index in indices:
            params = equal_samples[index]
            params_dict = retrieval_result.fit_info._interpret_param_array(params)
            t_p_profile.set_from_params_dict(profile_type, params_dict)
            temperature_arr.append(xp.cpu(t_p_profile.temperatures))

        plt.figure()
        if plot_samples:
            plt.plot(np.array(temperature_arr).T, profile_pressures / BAR_TO_PASCALS, color='b', alpha=0.25, zorder=2, label='samples') 
        if plot_1sigma_bounds:
            plt.fill_betweenx(profile_pressures / BAR_TO_PASCALS, np.percentile(temperature_arr, 16, axis=0),
                            np.percentile(temperature_arr, 84, axis=0), color='0.1', alpha=0.25, zorder=1, label='1$\\sigma$ bounds')  

        params_dict = retrieval_result.fit_info._interpret_param_array(retrieval_result.best_fit_params)
        t_p_profile.set_from_params_dict(profile_type, params_dict)
        plt.plot(xp.cpu(t_p_profile.temperatures), profile_pressures / BAR_TO_PASCALS, zorder=3, color='r', label='best fit')

        plt.yscale('log')   
        plt.ylim(min(profile_pressures / BAR_TO_PASCALS), max(profile_pressures / BAR_TO_PASCALS))
        plt.gca().invert_yaxis()               
        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure/bars")
        plt.legend()
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_retrieved_temp_profiles.png")


    def plot_retrieval_corner(self, retrieval_result, filename=None, **args):
        """
        Input a RetrievalResult object to make a corner plot for the 
        posteriors of the fitted parameters.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        if retrieval_result.retrieval_type == "dynesty":
            fig = corner.corner(retrieval_result.samples, weights=retrieval_result.weights,
                                range=[0.99] * retrieval_result.samples.shape[1],
                                show_titles=True,
                                labels=retrieval_result.fit_info.fit_param_names, **args)
        elif retrieval_result.retrieval_type == "pymultinest":
            fig = corner.corner(retrieval_result.equal_samples,
                                range=[0.99] * retrieval_result.equal_samples.shape[1],
                                show_titles=True,
                                labels=retrieval_result.fit_info.fit_param_names, **args)                
        elif retrieval_result.retrieval_type == "emcee":
            fig = corner.corner(retrieval_result.flatchain,
                                range=[0.99] * retrieval_result.flatchain.shape[1],
                                labels=retrieval_result.fit_info.fit_param_names, **args)
        else:
            assert(False)

        if filename is not None:
            fig.savefig(filename)


    def plot_retrieval_transit_spectrum(self, retrieval_result, prefix=None):
        """
        Input a RetrievalResult object to make a plot of the data,
        best fit transit model both at native resolution and data's resolution, 
        and a 1 sigma range for models.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        assert(retrieval_result.transit_bins is not None)

        plt.figure(figsize=(16,6))
        lower_spectrum = np.percentile(retrieval_result.random_transit_depths, 16, axis=0)
        upper_spectrum = np.percentile(retrieval_result.random_transit_depths, 84, axis=0)
        plt.fill_between(METRES_TO_UM * retrieval_result.best_fit_transit_dict["unbinned_wavelengths"],
                            lower_spectrum,
                            upper_spectrum,
                            color="#f2c8c4", zorder=2)            
        plt.plot(METRES_TO_UM * retrieval_result.best_fit_transit_dict["unbinned_wavelengths"],
                    retrieval_result.best_fit_transit_dict["unbinned_depths"] * 
                    retrieval_result.best_fit_transit_dict['unbinned_correction_factors'],
                    color='r', label="Calculated (unbinned)", zorder=3)
        plt.errorbar(METRES_TO_UM * retrieval_result.transit_wavelengths,
                        retrieval_result.transit_depths,
                        yerr = retrieval_result.transit_errors,
                        fmt='.', color='k', label="Observed", zorder=5)
        plt.scatter(METRES_TO_UM * retrieval_result.transit_wavelengths,
                    retrieval_result.best_fit_transit_depths,
                    color='b', label="Calculated (binned)", zorder=4)                        
                            
        plt.xlabel("Wavelength ($\mu m$)")
        plt.ylabel("Transit depth")
        plt.xscale('log')
        plt.tight_layout()
        plt.legend()
        if prefix is not None:
            plt.savefig(prefix + "_transit.png")


    def plot_retrieval_eclipse_spectrum(self, retrieval_result, prefix=None):
        """
        Input a RetrievalResult object to make a plot of the data,
        best fit eclipse model both at native resolution and data's resolution, 
        and a 1 sigma range for models.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        assert(retrieval_result.eclipse_bins is not None)
    
        plt.figure(figsize=(16,6))
        lower_spectrum = np.percentile(retrieval_result.random_eclipse_depths, 16, axis=0)
        upper_spectrum = np.percentile(retrieval_result.random_eclipse_depths, 84, axis=0)
        plt.fill_between(METRES_TO_UM * retrieval_result.best_fit_eclipse_dict["unbinned_wavelengths"],
                            lower_spectrum,
                            upper_spectrum,
                            color="#f2c8c4")
        plt.plot(METRES_TO_UM * retrieval_result.best_fit_eclipse_dict["unbinned_wavelengths"],
                    retrieval_result.best_fit_eclipse_dict["unbinned_eclipse_depths"],
                    alpha=0.4, color='r', label="Calculated (unbinned)")
        plt.errorbar(METRES_TO_UM * retrieval_result.eclipse_wavelengths,
                        retrieval_result.eclipse_depths,
                        yerr=retrieval_result.eclipse_errors,
                        fmt='.', color='k', label="Observed")
        plt.scatter(METRES_TO_UM * retrieval_result.eclipse_wavelengths,
                    retrieval_result.best_fit_eclipse_depths,
                    color='r', label="Calculated (binned)")
        plt.legend()
        plt.xlabel("Wavelength ($\mu m$)")
        plt.ylabel("Eclipse depth")
        plt.xscale('log')
        plt.tight_layout()
        plt.legend()
        if prefix is not None:
            plt.savefig(prefix + "_eclipse.png")


    def plot_optical_depth(self, depth_dict, prefix=None):
        """
        Input a depth dictionary created by the TransitDepthCalculator or EclipseDepthCalculator
        to plot optical depth as a function of wavelength and pressure.
        """
        plt.figure(figsize=(6,4))
        
        if 'tau_los' in depth_dict.keys():
            plt.contourf(depth_dict['unbinned_wavelengths'] * METRES_TO_UM, 
                         np.log10(0.5 * (depth_dict['P_profile'][1:] + depth_dict['P_profile'][:-1]) / BAR_TO_PASCALS), np.log10(depth_dict['tau_los'].T), cmap='magma_r')
            fname = '_transit'
        elif 'taus' in depth_dict.keys():
            plt.contourf(depth_dict['unbinned_wavelengths'] * METRES_TO_UM, 
                         np.log10(0.5 * (depth_dict['P_profile'][1:] + depth_dict['P_profile'][:-1]) / BAR_TO_PASCALS), np.log10(depth_dict['taus'].T), cmap='magma_r')
            fname = '_eclipse'
        else:
            print("Depth dictionary does not contain optical depth information.")
            assert(False)

        cbar = plt.colorbar(location='right')
        cbar.set_label('log (Optical depth)')
        plt.gca().invert_yaxis()
        plt.xlabel('Wavelength ($\\mu$m)')
        plt.ylabel('log (Pressure/bars)')
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + fname + "_optical_depth.png")


    def plot_eclipse_contrib_func(self, eclipse_depth_dict, log_scale=False, prefix=None):
        """
        Input an eclipse depth dictionary created by the EclipseDepthCalculator
        to plot emission contribution function as a function of wavelength and pressure.
        The log_scale parameter allows the user to toggle between plotting of the contribution 
        function in log or linear scale. 
        """
        assert('contrib' in eclipse_depth_dict.keys())

        if log_scale:
            contrib_func = np.log10(eclipse_depth_dict['contrib'].T)
            contrib_func[np.logical_or(np.isinf(contrib_func), contrib_func < -9.)] = np.nan
        else:
            contrib_func = eclipse_depth_dict['contrib'].T

        plt.figure(figsize=(6,4))
        plt.contourf(eclipse_depth_dict['unbinned_wavelengths'] * METRES_TO_UM, 
                         np.log10(0.5 * (eclipse_depth_dict['P_profile'][1:] + eclipse_depth_dict['P_profile'][:-1]) / BAR_TO_PASCALS), contrib_func, cmap='magma_r', vmin=np.nanmin(contrib_func), vmax=np.nanmax(contrib_func))

        cbar = plt.colorbar(location='right')
        if log_scale:cbar.set_label('log (Contribution function)')
        else:cbar.set_label('Contribution function')
        plt.gca().invert_yaxis()
        plt.xlabel('Wavelength ($\\mu$m)')
        plt.ylabel('log (Pressure/bars)')
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_eclipse_contrib_func.png")
        

    def plot_atm_abundances(self, atm_info, min_abund=1e-9, prefix=None):
        """
        Input a depth dictionary created by the TransitDepthCalculator or EclipseDepthCalculator
        or a dictionary outputed by AtmsophereSolver
        to plot abundance of different species (calculated for a given TP profile) as a function of pressure.
        """
        assert('atm_abundances' in atm_info.keys())
        abundances = atm_info['atm_abundances']

        plt.figure()
        for k in abundances.keys():
            if k == 'He' or k == 'H2' or k == 'H':
                continue
            if np.any(abundances[k] > min_abund):
                plt.loglog(abundances[k], atm_info['P_profile'] / BAR_TO_PASCALS, label=k)

        plt.gca().invert_yaxis()
        plt.xlim(min_abund,)
        plt.xlabel('Abundance ($n/n_{\\rm tot}$)')
        plt.ylabel('Pressure (bars)')
        plt.legend()
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_atm_abundances.png")
