from platon.constants import METRES_TO_UM
import matplotlib.pyplot as plt
import numpy as np
import corner
from astropy.io import ascii
import sys
import copy

from ._convert_clr_to_vmr import convert_clr_to_vmr

class RetrievalResult:    
    def __init__(self, results, retrieval_type="dynesty",
                 transit_bins=None, transit_depths=None, transit_errors=None,
                 eclipse_bins=None, eclipse_depths=None, eclipse_errors=None,
                 best_fit_transit_depths=None, best_fit_transit_dict=None,
                 best_fit_eclipse_depths=None, best_fit_eclipse_dict=None,
                 fit_info=None, samples_vmr_space = None, labels_vmr_space = None):
        
        self.results = results
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
            self.eclipse_chi_sqr = np.sum((eclipse_bins[:,0] + eclipse_bins[:,1])**2 / eclipse_errors**2)

        self.best_fit_transit_depths = best_fit_transit_depths
        self.best_fit_transit_dict = best_fit_transit_dict
        self.best_fit_eclipse_depths = best_fit_eclipse_depths
        self.best_fit_eclipse_dict = best_fit_eclipse_dict

        self.fit_info = fit_info
        self.__dict__.update(results)

        if "logz" in results:
            self.final_logz = results["logz"][-1]
            
        self.samples_vmr_space = samples_vmr_space
        self.labels_vmr_space = labels_vmr_space
        
        for i, name in enumerate(fit_info.fit_param_names):
            if name.split('_')[0] == 'clr':
                samples_vmr_space = np.copy(results.samples)
        
        if samples_vmr_space is not None:
            fit_labels_here = np.copy(fit_info.fit_param_names)
            inds_of_clrs = []
            for i, name in enumerate(fit_info.fit_param_names):
                if name.split('_')[0] == 'clr':
                    inds_of_clrs += [i]
                    fit_labels_here[i] = f"vmr_{name.split('_')[1]}"
            inds_of_clrs = np.array(inds_of_clrs)   
            
            for j,sample_set in enumerate(samples_vmr_space):
                clrs = []
                vmrs = []
                for i in inds_of_clrs:
                    clrs += [sample_set[i]]
                clrs = np.array(clrs)
                vmrs = np.log10(convert_clr_to_vmr(np.array(clrs)))
                for i in inds_of_clrs:
                    samples_vmr_space[j][i] = (np.array(vmrs[i-inds_of_clrs[0]]))
                
            self.samples_vmr_space = samples_vmr_space
            self.labels_vmr_space = fit_labels_here
                
                 
            
    def __repr__(self):
        return str(self.__dict__)

    def plot_corner(self, filename="multinest_corner.png", file_name_vmr = None, truths = None):
        plt.clf()
        if self.retrieval_type == "dynesty":
            new_samples = (self.samples) 
            new_labels = self.fit_info.fit_param_names
            
            if self.samples_vmr_space is not None:
                new_samples = (self.samples_vmr_space) 
                new_labels = self.labels_vmr_space
            
            if truths is not None:
                    truths = (truths)
            else: truths = None
            
            fig = corner.corner(new_samples, weights=self.weights,
                            range=[0.99] * self.samples.shape[1],
                            labels=new_labels, truths = truths, truth_color = 'cornflowerblue')
            
            if file_name_vmr is not None:
                fig.savefig(f'{file_name_vmr}', transparency = False)
                
            else:
                fig.savefig(filename, transparency = False)
            
    
        elif self.retrieval_type == "emcee":
            fig = corner.corner(self.flatchain,
                                range=[0.99] * self.flatchain.shape[1],
                                labels=self.fit_info.fit_param_names)
            fig.savefig(filename, transparency = False)
        else:
            assert(False)
            
    def plot_spectrum(self, prefix="best_fit", true_model = None):
        plt.clf()
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
                             lower_spectrum * 1e6,
                             upper_spectrum * 1e6,
                             color="#f2c8c4")
            plt.plot(METRES_TO_UM * self.best_fit_eclipse_dict["unbinned_wavelengths"],
                     self.best_fit_eclipse_dict["unbinned_eclipse_depths"] * 1e6,
                     alpha=0.4, color='r', label="Calculated (unbinned)")
            plt.errorbar(METRES_TO_UM * self.eclipse_wavelengths,
                         self.eclipse_depths * 1e6,
                         yerr=self.eclipse_errors * 1e6,
                         fmt='.', color='k', label="Observed")
            plt.scatter(METRES_TO_UM * self.eclipse_wavelengths,
                        self.best_fit_eclipse_depths * 1e6,
                        color='r', label="Calculated (binned)")
            if true_model is not None:
                true_model_data = ascii.read(str(true_model))
                true_eclipse_wavelengths = np.array(true_model_data['Wavelength']) #m
                true_eclipse_depths = np.array(true_model_data['Fp/Fs'])
                plt.plot(METRES_TO_UM * true_eclipse_wavelengths,
                     true_eclipse_depths * 1e6,
                     alpha=0.8, color = 'k', linestyle = '--', label="Truth")
            plt.xlim(np.min(METRES_TO_UM * self.best_fit_eclipse_dict["unbinned_wavelengths"]), np.max(METRES_TO_UM * self.best_fit_eclipse_dict["unbinned_wavelengths"]))
            plt.legend()
            plt.xlabel("Wavelength ($\mu m$)")
            plt.ylabel("Eclipse depth [ppm]")
            # plt.xscale('log')
            plt.tight_layout()
            plt.legend()
            plt.savefig(prefix + "_eclipse.png")
            plt.close()

