import numpy as np

class RetrievalResult:
    def __init__(self, results, retrieval_type, best_fit_params,
                 transit_bins=None, transit_depths=None, transit_errors=None,
                 eclipse_bins=None, eclipse_depths=None, eclipse_errors=None,
                 best_fit_transit_depths=None, best_fit_transit_dict=None,
                 best_fit_eclipse_depths=None, best_fit_eclipse_dict=None,
                 fit_info=None, divisors=None, labels=None):

        self.best_fit_params = best_fit_params
        self.retrieval_type = retrieval_type
        self.transit_bins = transit_bins
        self.transit_depths = transit_depths
        self.transit_errors = transit_errors

        if transit_bins is not None:
            transit_bins = np.array(transit_bins)
            self.transit_wavelengths = (transit_bins[:,0] + transit_bins[:,1]) / 2
            self.transit_chi_sqr = np.sum((transit_depths - best_fit_transit_depths)**2 / transit_errors**2)
            print("Transit chi sqr", self.transit_chi_sqr)
            
        
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
    
