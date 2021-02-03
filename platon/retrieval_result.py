from platon.constants import METRES_TO_UM
import matplotlib.pyplot as plt
import numpy as np
import corner

class RetrievalResult:    
    def __init__(self, results, retrieval_type="dynesty",                 
                 flux_bins=None, fluxes=None, flux_errors=None,
                 best_fit_fluxes=None, best_fit_flux_dict=None,
                 fit_info=None):
        
        self.results = results
        self.retrieval_type = retrieval_type
        self.flux_bins = np.array(flux_bins)
        self.fluxes = fluxes
        self.flux_errors = flux_errors

        self.flux_wavelengths = (self.flux_bins[:,0] + self.flux_bins[:,1]) / 2
        self.flux_chi_sqr = np.sum((self.flux_bins[:,0] + self.flux_bins[:,1])**2 / flux_errors**2)

        self.best_fit_fluxes = best_fit_fluxes
        self.best_fit_flux_dict = best_fit_flux_dict
        self.fit_info = fit_info
        self.__dict__.update(dict(results))

        if "logz" in results:
            self.final_logz = results["logz"][-1]
            
    def __repr__(self):
        return str(self.__dict__)

    def plot_corner(self, filename="multinest_corner.png"):
        plt.clf()
        if self.retrieval_type == "dynesty":
            fig = corner.corner(self.samples, weights=self.weights,
                          range=[0.99] * self.samples.shape[1],
                          labels=self.fit_info.fit_param_names)
            fig.savefig(filename)
        elif self.retrieval_type == "emcee":
            fig = corner.corner(self.flatchain,
                                range=[0.99] * self.flatchain.shape[1],
                                labels=self.fit_info.fit_param_names)
            fig.savefig(filename)
        else:
            assert(False)
            
    def plot_spectrum(self, prefix="best_fit"):
        plt.clf()

        plt.figure(1, figsize=(16,6))
        lower_spectrum = np.percentile(self.random_fluxes, 16, axis=0)
        upper_spectrum = np.percentile(self.random_fluxes, 84, axis=0)
        plt.fill_between(METRES_TO_UM * self.best_fit_flux_dict["unbinned_wavelengths"],
                         lower_spectrum,
                         upper_spectrum,
                         color="#f2c8c4")            
        plt.plot(METRES_TO_UM * self.best_fit_flux_dict["unbinned_wavelengths"],
                 self.best_fit_flux_dict["unbinned_fluxes"],
                 alpha=0.4, color='r', label="Calculated (unbinned)")
        plt.errorbar(METRES_TO_UM * self.flux_wavelengths,
                     self.fluxes,
                     yerr = self.flux_errors,
                     fmt='.', color='k', label="Observed")
        plt.scatter(METRES_TO_UM * self.flux_wavelengths,
                    self.best_fit_fluxes,
                    color='b', label="Calculated (binned)")                                
        plt.xlabel("Wavelength ($\mu m$)")
        plt.ylabel("Flux (W/m^3)")
        plt.xscale('log')
        plt.tight_layout()
        plt.legend()
        plt.savefig(prefix + "_flux.pdf")        
