class RetrievalResult:    
    def __init__(self, results, retrieval_type="dynesty",
                 transit_bins=None, transit_depths=None, transit_errors=None,
                 eclipse_bins=None, eclipse_depths=None, eclipse_errors=None,
                 best_fit_transit_depths=None, best_fit_eclipse_depths=None,
                 fit_info=None):
        self.transit_bins = transit_bins
        self.transit_depths = transit_depths
        self.transit_errors = transit_errors

        self.eclipse_bins = eclipse_bins
        self.eclipse_depths = eclipse_depths
        self.eclipse_errors = eclipse_errors

        self.best_fit_transit_depths = best_fit_transit_depths
        self.best_fit_eclipse_depths = best_fit_eclipse_depths

        self.fit_info = fit_info
        self.__dict__.update(dict(results))
                 
            
    def __repr__(self):
        return str(self.__dict__)
