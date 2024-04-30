import matplotlib.pyplot as plt
import numpy as np
import corner
from .constants import METRES_TO_UM
import matplotlib

class RetrievalResult:
    def __init__(self, results, retrieval_type="dynesty",
                 transit_bins=None, transit_depths=None, transit_errors=None,
                 eclipse_bins=None, eclipse_depths=None, eclipse_errors=None,
                 best_fit_transit_depths=None, best_fit_transit_dict=None,
                 best_fit_eclipse_depths=None, best_fit_eclipse_dict=None,
                 fit_info=None, divisors=None, labels=None):
        
        self.retrieval_type = retrieval_type
        self.transit_bins = transit_bins
        self.transit_depths = transit_depths
        self.transit_errors = transit_errors

        if transit_bins is not None:
            transit_bins = np.array(transit_bins)
            self.transit_wavelengths = (transit_bins[:,0] + transit_bins[:,1]) / 2
            self.transit_chi_sqr = np.sum((transit_depths - best_fit_transit_depths)**2 / transit_errors**2)
            
        
        self.eclipse_bins = eclipse_bins
        self.eclipse_depths = eclipse_depths
        self.eclipse_errors = eclipse_errors

        if eclipse_bins is not None:
            eclipse_bins = np.array(eclipse_bins)
            self.eclipse_wavelengths = (eclipse_bins[:,0] + eclipse_bins[:,1]) / 2
            self.eclipse_chi_sqr = np.sum((eclipse_depths - best_fit_eclipse_depths)**2 / eclipse_errors**2)

        self.best_fit_transit_depths = best_fit_transit_depths
        self.best_fit_transit_dict = best_fit_transit_dict
        self.best_fit_eclipse_depths = best_fit_eclipse_depths
        self.best_fit_eclipse_dict = best_fit_eclipse_dict

        self.fit_info = fit_info
        self.__dict__.update(results)

        if "logz" in results:
            self.final_logz = results["logz"][-1]

        self.divisors = divisors
        self.labels = labels
                 
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def items(self):
        return list(self.__dict__.items())

    def __repr__(self):
        return str(self.__dict__)
    
    def plot_corner(self, filename="multinest_corner.png"):       
        plt.clf()
        if self.retrieval_type == "dynesty":
            fig = corner.corner(self.samples/self.divisors,
                                weights=self.weights,
                                range=[0.99] * self.samples.shape[1],
                                labels=self.labels,
                                smooth=0.8, show_titles=True)
            fig.savefig(filename)
        elif self.retrieval_type == "emcee":
            fig = corner.corner(self.flatchain / self.divisors,
                                range=[0.99] * self.flatchain.shape[1],
                                labels=self.labels,
                                smooth=0.8, show_titles=True)
            fig.savefig(filename)
        else:
            assert(False)
            
    def plot_spectrum(self, prefix="best_fit"):
        plt.clf()
        N_trans = 0 #will update in the if
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
            cmap = plt.cm.get_cmap("viridis")
            N_trans = len(self.transit_wavelengths)
            norm = matplotlib.colors.Normalize(vmin=np.min(self.loos[:N_trans]), vmax=np.max(self.loos[:N_trans]))
            for i in range(N_trans):
                plt.errorbar(METRES_TO_UM * self.transit_wavelengths[i],
                             self.transit_depths[i],
                             yerr = self.transit_errors[i],
                             fmt='.', color=cmap(norm(self.loos[i])))
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                ax=plt.gca(),
                                label="Leave-one-out log predictive density")
            
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
                     alpha=0.4, color='r', label="Calculated (unbinned)")

            cmap = plt.cm.get_cmap("viridis")
            norm = matplotlib.colors.Normalize(vmin=np.min(self.loos[N_trans:]), vmax=np.max(self.loos[N_trans:]))
            for i in range(len(self.eclipse_bins)):
                plt.errorbar(METRES_TO_UM * self.eclipse_wavelengths[i],
                             self.eclipse_depths[i],
                             yerr=self.eclipse_errors[i],
                             fmt='.', color=cmap(norm(self.loos[N_trans+i])))
            
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                ax=plt.gca(),
                                label="Leave-one-out log predictive density")
            
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
