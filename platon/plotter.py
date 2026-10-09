import matplotlib.pyplot as plt
import numpy as np
import corner
import dynesty
from .constants import METRES_TO_UM, BAR_TO_PASCALS
from .retrieval_result import RetrievalResult
from .TP_profile import Profile
from .terminator import label_by_temperature

default_style = ['default',
    {   'font.size': 12,
        'xtick.top': True,
        'xtick.direction': 'out',
        'ytick.right': True,
        'ytick.direction': 'out',
        }]
plt.style.use(default_style)

class Plotter:
    """Plotting functions for PLATON results.  All methods are class
    methods, so call them directly, e.g. Plotter.plot_retrieval_corner(result)."""

    @classmethod
    def plot_retrieval_TP_profiles(cls, retrieval_result, plot_samples=False,
                                   plot_1sigma_bounds=True, num_samples=100,
                                   prefix=None, which=None):
        """
        Input a RetrievalResult object to make a plot of the best fit temperature profile
        and 1 sigma bounds for the profile and/or plot samples of the temperature profile.

        Parameters
        ----------
        which : str, optional
            "transit" to plot the terminator profile used for transit depths,
            "eclipse" to plot the dayside profile used for eclipse depths.
            Defaults to "eclipse" if eclipse data were fit, "transit"
            otherwise.

        The posterior samples stored in retrieval_result
        (random_transit_TP_profiles / random_eclipse_TP_profiles) are used
        when available; otherwise the profiles are recomputed from the
        posterior samples of the fit parameters.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        if which is None:
            which = "eclipse" if retrieval_result.eclipse_depths is not None \
                else "transit"
        if which not in ("transit", "eclipse"):
            raise ValueError("which must be 'transit' or 'eclipse'")

        fit_info = retrieval_result.fit_info
        best_params_dict = fit_info._interpret_param_array(
            retrieval_result.best_fit_params)

        terminator = None
        if which == "transit":
            terminator_param = fit_info.all_params.get("transit_terminator")
            terminator = None if terminator_param is None else \
                terminator_param.best_guess

        plt.figure()
        if terminator is not None:
            models = [terminator.from_params(params_dict) for params_dict in
                      cls._random_params_dicts(retrieval_result, num_samples)]
            cold_temperatures = np.array(
                [model.cold.profile.temperatures for model in models])
            hot_temperatures = np.array(
                [model.hot.profile.temperatures for model in models])
            pressure_bars = terminator.cold.profile.pressures / BAR_TO_PASCALS

            if plot_samples:
                plt.plot(cold_temperatures.T, pressure_bars, color="C0",
                         alpha=0.12, zorder=1)[0].set_label("cold samples")
                plt.plot(hot_temperatures.T, pressure_bars, color="C3",
                         alpha=0.12, zorder=1)[0].set_label("hot samples")
            if plot_1sigma_bounds:
                cls._plot_1sigma_bounds(cold_temperatures, pressure_bars,
                                        color="C0", label="cold 1$\\sigma$")
                cls._plot_1sigma_bounds(hot_temperatures, pressure_bars,
                                        color="C3", label="hot 1$\\sigma$")

            best = terminator.from_params(best_params_dict)
            plt.plot(best.cold.profile.temperatures, pressure_bars,
                     color="C0", label="cold best fit")
            plt.plot(best.hot.profile.temperatures, pressure_bars,
                     color="C3", label="hot best fit")
            cls._finish_TP_plot(pressure_bars, prefix)
            return

        # 1-D profile: prefer the samples stored during the retrieval
        if which == "transit":
            stored = getattr(retrieval_result, "random_transit_TP_profiles",
                             None)
            best_dict = retrieval_result.best_fit_transit_dict
            profile_type = fit_info.all_params.get("transit_profile_type")
            profile_type = "isothermal" if profile_type is None \
                else profile_type.best_guess
            suffix = "_transit"
        else:
            stored = getattr(retrieval_result, "random_eclipse_TP_profiles",
                             None)
            best_dict = retrieval_result.best_fit_eclipse_dict
            profile_type = fit_info.all_params["profile_type"].best_guess
            suffix = ""

        if stored is not None and len(stored) > 0:
            stored = np.asarray(stored)[:num_samples]
            profile_pressures = stored[0, 0]
            temperature_arr = stored[:, 1]
        else:
            profiles = [
                Profile.from_params_dict(profile_type, params_dict, suffix=suffix)
                for params_dict in cls._random_params_dicts(
                    retrieval_result, num_samples)]
            profile_pressures = profiles[0].pressures
            temperature_arr = np.array([p.temperatures for p in profiles])

        if best_dict is not None and "full_TP_profile" in best_dict:
            best_pressures, best_temperatures = best_dict["full_TP_profile"]
        else:
            t_p_profile = Profile.from_params_dict(
                profile_type, best_params_dict, suffix=suffix)
            best_pressures = t_p_profile.pressures
            best_temperatures = t_p_profile.temperatures

        pressure_bars = profile_pressures / BAR_TO_PASCALS
        if plot_samples:
            lines = plt.plot(temperature_arr.T, pressure_bars, color='b',
                             alpha=0.25, zorder=2)
            lines[0].set_label('samples')
        if plot_1sigma_bounds:
            cls._plot_1sigma_bounds(temperature_arr, pressure_bars,
                                    color='0.1', zorder=1,
                                    label='1$\\sigma$ bounds')

        plt.plot(best_temperatures, best_pressures / BAR_TO_PASCALS,
                 zorder=3, color='r', label='best fit')
        cls._finish_TP_plot(pressure_bars, prefix)

    @classmethod
    def _random_params_dicts(cls, retrieval_result, num_samples):
        """Parameter dictionaries of num_samples random posterior samples."""
        equal_samples = cls._get_equal_samples(retrieval_result)
        indices = np.random.choice(len(equal_samples), num_samples)
        return [retrieval_result.fit_info._interpret_param_array(
            equal_samples[i]) for i in indices]

    @staticmethod
    def _plot_1sigma_bounds(temperatures, pressure_bars, **kwargs):
        """Shades the 16th to 84th percentile of the temperature at each
        pressure."""
        plt.fill_betweenx(
            pressure_bars,
            np.percentile(temperatures, 16, axis=0),
            np.percentile(temperatures, 84, axis=0),
            alpha=0.25, **kwargs)

    @staticmethod
    def _finish_TP_plot(pressure_bars, prefix):
        plt.yscale("log")
        plt.ylim(pressure_bars.min(), pressure_bars.max())
        plt.gca().invert_yaxis()
        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure/bars")
        plt.legend()
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_retrieved_temp_profiles.png")

    @staticmethod
    def _get_equal_samples(retrieval_result):
        equal_samples = getattr(retrieval_result, "equal_samples", None)
        if equal_samples is not None:
            return equal_samples

        # Results saved by older versions of PLATON, which only stored
        # equally weighted samples for MultiNest
        if retrieval_result.retrieval_type in ("dynesty", "nautilus"):
            equal_samples = dynesty.utils.resample_equal(
                retrieval_result.samples, retrieval_result.weights)
            np.random.shuffle(equal_samples)
            return equal_samples
        if retrieval_result.retrieval_type == "emcee":
            return np.copy(retrieval_result.flatchain)
        raise ValueError("Unknown retrieval type: {}".format(
            retrieval_result.retrieval_type))

    @classmethod
    def plot_retrieval_corner(cls, retrieval_result, filename=None, **args):
        """
        Input a RetrievalResult object to make a corner plot for the
        posteriors of the fitted parameters.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        # Nested samplers have weighted samples, which are more precise than
        # equally weighted samples drawn from them
        if retrieval_result.retrieval_type in ("dynesty", "nautilus"):
            samples = retrieval_result.samples
            weights = retrieval_result.weights
        else:
            samples = cls._get_equal_samples(retrieval_result)
            weights = None

        # Two-sector parameters are shown as cold/hot by temperature
        labels, samples = label_by_temperature(retrieval_result.fit_info, samples)
        corner_args = dict(range=[0.99] * samples.shape[1], show_titles=True,
                           labels=labels)
        corner_args.update(args)
        fig = corner.corner(samples, weights=weights, **corner_args)

        if filename is not None:
            fig.savefig(filename)

    @classmethod
    def plot_retrieval_transit_spectrum(cls, retrieval_result, prefix=None):
        """
        Input a RetrievalResult object to make a plot of the data,
        best fit transit model both at native resolution and data's resolution,
        and a 1 sigma range for models.
        """
        cls._plot_retrieval_spectrum(retrieval_result, "transit", prefix)

    @classmethod
    def plot_retrieval_eclipse_spectrum(cls, retrieval_result, prefix=None):
        """
        Input a RetrievalResult object to make a plot of the data,
        best fit eclipse model both at native resolution and data's resolution,
        and a 1 sigma range for models.
        """
        cls._plot_retrieval_spectrum(retrieval_result, "eclipse", prefix)

    @staticmethod
    def _plot_retrieval_spectrum(retrieval_result, kind, prefix):
        """Plots the observed and best fit spectra, where kind is "transit"
        or "eclipse"."""
        assert(isinstance(retrieval_result, RetrievalResult))
        assert(retrieval_result[kind + "_bins"] is not None)

        best_fit_dict = retrieval_result["best_fit_{}_dict".format(kind)]
        unbinned_wavelengths = METRES_TO_UM * best_fit_dict["unbinned_wavelengths"]
        unbinned_depths = best_fit_dict[
            "unbinned_depths" if kind == "transit" else "unbinned_eclipse_depths"]
        random_depths = retrieval_result["random_{}_depths".format(kind)]
        wavelengths = METRES_TO_UM * retrieval_result[kind + "_wavelengths"]
        depths = retrieval_result[kind + "_depths"]
        # The pointwise LOO scores list the transit points, then the eclipse
        # points
        loos = retrieval_result["loos"]
        loos = loos[:len(depths)] if kind == "transit" else loos[-len(depths):]

        plt.figure(figsize=(16,6))
        plt.fill_between(unbinned_wavelengths,
                         np.percentile(random_depths, 16, axis=0),
                         np.percentile(random_depths, 84, axis=0),
                         color="#f2c8c4", zorder=2)
        plt.plot(unbinned_wavelengths, unbinned_depths, color='r',
                 label="Calculated (unbinned, unshifted)", zorder=3)
        plt.errorbar(wavelengths, depths,
                     yerr=retrieval_result[kind + "_errors"],
                     fmt='.', color='k', label="Observed", zorder=5)
        points = plt.scatter(
            wavelengths, depths, c=loos, cmap="viridis",
            s=25, edgecolors='k', linewidths=0.5,
            label="Observed", zorder=6)
        plt.colorbar(points, label="LOO log predictive density", pad=0.01)
        plt.scatter(wavelengths,
                    retrieval_result["best_fit_{}_depths".format(kind)],
                    color='b', label="Calculated (binned)", zorder=4)

        plt.xlabel("Wavelength ($\\mu m$)")
        plt.ylabel(kind.capitalize() + " depth")
        plt.xscale('log')
        plt.tight_layout()
        plt.legend()
        if prefix is not None:
            plt.savefig("{}_{}.png".format(prefix, kind))

    @staticmethod
    def _log_mid_pressure_bars(info_dict):
        """log10 of the pressures (in bars) midway between the layers of an
        info_dict's T/P profile"""
        P_profile = info_dict['P_profile']
        return np.log10(0.5 * (P_profile[1:] + P_profile[:-1]) / BAR_TO_PASCALS)

    @staticmethod
    def _finish_wavelength_pressure_plot(colorbar_label, filename):
        cbar = plt.colorbar(location='right')
        cbar.set_label(colorbar_label)
        plt.gca().invert_yaxis()
        plt.xlabel('Wavelength ($\\mu$m)')
        plt.ylabel('log (Pressure/bars)')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)

    @classmethod
    def plot_optical_depth(cls, depth_dict, prefix=None):
        """
        Input a depth dictionary created by the TransitDepthCalculator or EclipseDepthCalculator
        to plot optical depth as a function of wavelength and pressure.
        """
        if 'tau_los' in depth_dict:
            taus, kind = depth_dict['tau_los'], 'transit'
        elif 'taus' in depth_dict:
            taus, kind = depth_dict['taus'], 'eclipse'
        else:
            raise ValueError(
                "Depth dictionary does not contain optical depth information.")

        plt.figure(figsize=(6,4))
        plt.contourf(depth_dict['unbinned_wavelengths'] * METRES_TO_UM,
                     cls._log_mid_pressure_bars(depth_dict), np.log10(taus.T),
                     cmap='magma_r')
        cls._finish_wavelength_pressure_plot(
            'log (Optical depth)',
            None if prefix is None else
            "{}_{}_optical_depth.png".format(prefix, kind))

    @classmethod
    def plot_contrib_func(cls, info_dict, log_scale=False, prefix=None):
        """
        Input an info_dict created by the TransitDepthCalculator or EclipseDepthCalculator
        to plot emission contribution function as a function of wavelength and pressure.
        The log_scale parameter allows the user to toggle between plotting of the contribution
        function in log or linear scale.
        """
        assert('contrib' in info_dict)

        if log_scale:
            contrib_func = np.log10(info_dict['contrib'].T)
            contrib_func[np.logical_or(np.isinf(contrib_func), contrib_func < -9.)] = np.nan
        else:
            contrib_func = info_dict['contrib'].T

        plt.figure(figsize=(6,4))
        plt.contourf(info_dict['unbinned_wavelengths'] * METRES_TO_UM,
                     cls._log_mid_pressure_bars(info_dict), contrib_func,
                     cmap='magma_r', vmin=np.nanmin(contrib_func),
                     vmax=np.nanmax(contrib_func))
        cls._finish_wavelength_pressure_plot(
            'log (Contribution function)' if log_scale else 'Contribution function',
            None if prefix is None else prefix + "_contrib_func.png")

    @classmethod
    def plot_atm_abundances(cls, atm_info, min_abund=1e-9, prefix=None):
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
