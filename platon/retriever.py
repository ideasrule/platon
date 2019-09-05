from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import emcee
import copy

from .transit_depth_calculator import TransitDepthCalculator
from .fit_info import FitInfo
from .constants import METRES_TO_UM
from ._params import _UniformParam
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .combined_retriever import CombinedRetriever

class Retriever:
    def __init__(self):
        self.combined_retriever = CombinedRetriever()
        
    def run_emcee(self, wavelength_bins, depths, errors, fit_info, nwalkers=50,
                  nsteps=1000, include_condensation=True,
                  plot_best=False):
        '''Runs affine-invariant MCMC to retrieve atmospheric parameters.

        Parameters
        ----------
        wavelength_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        errors : array_like, length N
            Errors on the aforementioned transit depths
        fit_info : :class:`.FitInfo` object
            Tells the method what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        nwalkers : int, optional
            Number of walkers to use
        nsteps : int, optional
            Number of steps that the walkers should walk for
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        plot_best : bool, optional
            If True, plots the best fit model with the data

        Returns
        -------
        result : EnsembleSampler object
            This returns emcee's EnsembleSampler object.  The most useful
            attributes in this item are result.chain, which is a (W x S X P)
            array where W is the number of walkers, S is the number of steps,
            and P is the number of parameters; and result.lnprobability, a
            (W x S) array of log probabilities.  For your convenience, this
            object also contains result.flatchain, which is a (WS x P) array
            where WS = W x S is the number of samples; and
            result.flatlnprobability, an array of length WS
        '''
        return self.combined_retriever.run_emcee(
            wavelength_bins, depths, errors, None, None, None,
            fit_info, nwalkers, nsteps, include_condensation, plot_best)
                                          

    def run_multinest(self, wavelength_bins, depths, errors, fit_info,
                      include_condensation=True, plot_best=False,
                      maxiter=None, maxcall=None, nlive=100,
                      **dynesty_kwargs):
        '''Runs nested sampling to retrieve atmospheric parameters.

        Parameters
        ----------
        wavelength_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        errors : array_like, length N
            Errors on the aforementioned transit depths
        fit_info : :class:`.FitInfo` object
            Tells us what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        plot_best : bool, optional
            If True, plots the best fit model with the data
        nlive : int
            Number of live points to use for nested sampling
        **dynesty_kwargs : keyword arguments to pass to dynesty's NestedSampler

        Returns
        -------
        result : Result object
            This returns 'results' of the NestedSampler object.  It is
            dictionary-like and has many useful items.  For example,
            result.samples (or alternatively, result["samples"]) are the
            parameter values of each sample, result.weights contains the
            weights, and result.logl contains the log likelihoods.  result.logz
            is the natural logarithm of the evidence.
        '''
        return self.combined_retriever.run_multinest(
            wavelength_bins, depths, errors, None, None, None, fit_info,
            include_condensation, plot_best, maxiter, maxcall,
            nlive=nlive, **dynesty_kwargs)
            

    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T, logZ=0, CO_ratio=0.53,
                             log_cloudtop_P=np.inf, log_scatt_factor=0,
                             scatt_slope=4, error_multiple=1, T_star=None,
                             T_spot=None, spot_cov_frac=None,frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size =-6,
                             part_size_std = 0.5, ri = None):
        '''Get a :class:`.FitInfo` object filled with best guess values.  A few
        parameters are required, but others can be set to default values if you
        do not want to specify them.  All parameters are in SI.
        For information on the parameters, see the documentation for 
        :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`

        Returns
        -------
        fit_info : :class:`.FitInfo` object
            This object is used to indicate which parameters to fit for, which
            to fix, and what values all parameters should take.'''

        fit_info = FitInfo(locals().copy())
        return fit_info
