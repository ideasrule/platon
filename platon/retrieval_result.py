from platon.constants import METRES_TO_UM
import matplotlib.pyplot as plt
import numpy as np
import corner

class RetrievalResult:    
    def __init__(self, results, retrieval_type="dynesty",
                 transit_bins=None, transit_depths=None, transit_errors=None,
                 eclipse_bins=None, eclipse_depths=None, eclipse_errors=None,
                 best_fit_transit_depths=None, best_fit_transit_dict=None,
                 best_fit_eclipse_depths=None, best_fit_eclipse_dict=None,
                 fit_info=None):
        self.results = results
        self.retrieval_type = retrieval_type
        self.transit_bins = transit_bins
        self.transit_depths = transit_depths
        self.transit_errors = transit_errors

        if transit_bins is not None:
            self.transit_wavelengths = (transit_bins[:,0] + transit_bins[:,1]) / 2
            self.transit_chi_sqr = np.sum((transit_depths - best_fit_transit_depths)**2 / transit_errors**2)
            
        
        self.eclipse_bins = eclipse_bins
        self.eclipse_depths = eclipse_depths
        self.eclipse_errors = eclipse_errors

        if eclipse_bins is not None:
            self.eclipse_wavelengths = (eclipse_bins[:,0] + eclipse_bins[:,1]) / 2
            self.eclipse_chi_sqr = np.sum((eclipse_bins[:,0] + eclipse_bins[:,1])**2 / eclipse_errors**2)

        self.best_fit_transit_depths = best_fit_transit_depths
        self.best_fit_transit_dict = best_fit_transit_dict
        self.best_fit_eclipse_depths = best_fit_eclipse_depths
        self.best_fit_eclipse_dict = best_fit_eclipse_dict

        self.fit_info = fit_info
        self.__dict__.update(dict(results))

        self.final_logz = self.logz[-1]
                 
            
    def __repr__(self):
        return str(self.__dict__)

    def plot_corner(self, filename="multinest_corner.png"):
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
        if self.transit_bins is not None:    
            plt.figure(1, figsize=(16,6))
            lower_spectrum = np.percentile(self.random_transit_depths, 16, axis=0)
            upper_spectrum = np.percentile(self.random_transit_depths, 84, axis=0)
            plt.fill_between(METRES_TO_UM * self.best_fit_transit_dict["unbinned_wavelengths"],
                             lower_spectrum,
                             upper_spectrum,
                             color="#f2c8c4")            
            plt.plot(METRES_TO_UM * self.best_fit_transit_dict["unbinned_wavelengths"],
                     self.best_fit_transit_dict["unbinned_depths"],
                     alpha=0.4, color='r', label="Calculated (unbinned)")
            plt.errorbar(METRES_TO_UM * self.transit_wavelengths,
                         self.transit_depths,
                         yerr = self.transit_errors,
                         fmt='.', color='k', label="Observed")
            plt.scatter(METRES_TO_UM * self.transit_wavelengths,
                        self.best_fit_transit_depths,
                        color='b', label="Calculated (binned)")                        
                             
            plt.xlabel("Wavelength ($\mu m$)")
            plt.ylabel("Transit depth")
            plt.xscale('log')
            plt.tight_layout()
            plt.legend()
            plt.savefig(prefix + "_transit.pdf")

        if self.eclipse_bins is not None:
            plt.figure(2, figsize=(16,6))
            lower_spectrum = np.percentile(self.random_eclipse_depths, 16, axis=0)
            upper_spectrum = np.percentile(self.random_eclipse_depths, 84, axis=0)
            plt.fill_between(METRES_TO_UM * self.best_fit_eclipse_dict["unbinned_wavelengths"],
                             lower_spectrum,
                             upper_spectrum,
                             color="#f2c8c4") 
            plt.plot(METRES_TO_UM * self.best_fit_eclipse_dict["unbinned_wavelengths"],
                     self.best_fit_eclipse_dict["unbinned_eclipse_depths"],
                     alpha=0.4, color='b', label="Calculated (unbinned)")
            plt.errorbar(METRES_TO_UM * self.eclipse_wavelengths,
                         self.eclipse_depths,
                         yerr=self.eclipse_errors,
                         fmt='.', color='k', label="Observed")
            plt.scatter(METRES_TO_UM * self.eclipse_wavelengths,
                        self.best_fit_eclipse_depths,
                        color='r', label="Calculated (binned)")
            plt.legend()
            plt.xlabel("Wavelength ($\mu m$)")
            plt.ylabel("Eclipse depth")
            plt.xscale('log')
            plt.tight_layout()
            plt.legend()
            plt.savefig(prefix + "_eclipse.pdf")
