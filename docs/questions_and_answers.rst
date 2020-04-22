Questions & Answers
*******************

This document describes niche use cases that the Quick Start does not cover.
For typical usage patterns, consult the files in examples/ and the Quick Start,
in that order.

* **What physics does PLATON take into account?**

  We account for gas absorption, collisional absorption, an opaque
  cloud deck, and scattering with user-specified slope and amplitude
  (or Rayleigh, if not specified).  H- bound-free and free-free absorption
  is not enabled by default, but can be turned on by passing add_H_minus_absorption=True to compute_depths.  34 chemical species are included
  in our calculations, namely the ones listed in data/species_info.
  The abundances of these species were calculated using GGchem for a
  grid of metallicity, C/O ratio, temperature, and pressure, assuming
  equilibrium chemistry with or without condensation.  Condensation can be
  toggled using include_condensation=True/False.  Metallicity ranges from 0.1-1000x solar, C/O
  ratio from 0.05 to 2, temperature from 200 to 3000 K, and pressure
  from 10^-4 to 10^8 Pa.  If you wander outside these limits, PLATON
  will throw a ValueError.
  
* **How do I specify custom abundances and T/P profiles?**
  
  By example: ::
    
    from platon.abundance_getter import AbundanceGetter
    from platon.transit_depth_calculator import TransitDepthCalculator
    
    _, pressures, temperatures = np.loadtxt("t_p_1200K.dat", skiprows=1, unpack=True)

    # These files are found in examples/custom_abundances.  They are equivalent
    # to the ExoTransmit EOS files, except that COS is renamed to OCS.  They provide
    # the abundance at every pressure and temperature grid point.  To create your
    # own, see the documentation for custom_abundances in
    #:func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`    
    abundances = AbundanceGetter.from_file("abund_1Xsolar_cond.dat")

  Alternatively, one can set vertically constant abundances for some species
  by getting the equilibrium abundances, then modifying them ::

    from platon.abundance_getter import AbundanceGetter
    getter = AbundanceGetter()
    # Solar logZ and C/O ratio. Modify as required.
    abundances = getter.get(0, 0.53)

    # Zero out CO.  (Note that if CO is a major component, you should probably
    # renormalize the abundances of other species so that they add up to 1.)    
    abundances["CO"] *= 0

    # Set CH4 abundance to a constant throughout the atmosphere
    abundances["CH4"] *= 0
    abundances["CH4"] += 1e-5

    
* **How do I do check what effect a species has on the transit spectrum?**
  Use the method above to zero out abundances of one species at a time.  
  Then call compute_depths with logZ and CO_ratio set to None: ::

    calculator.compute_depths(star_radius, planet_mass, planet_radius, temperature,
    logZ=None, CO_ratio=None, custom_abundances=abundances)

  Alternatively, you can delete absorption coefficients from PLATON_DIR/platon/data/Absorption,
  which has the effect of zeroing the opacity of those molecules.


* **Which parameters are supported in retrieval?**
  See the documentation for :func:`~platon.combined_retriever.CombinedRetriever.get_default_fit_info`.  All arguments to this method are possible fit parameters.  However, we
  recommend not fitting for T_star, as it has a very small effect on the result
  to begin with.  Mp and Rs are usually measured to greater precision than you
  can achieve in a fit, but we recommend fitting them with Gaussian priors to
  take into account the measurement errors.

* **Should I use run_multinest, or run_emcee?**
  
  That depends on whether you like nested sampling or MCMC!  We recommend nested sampling because it handles multimodal distributions more robustly, and because it has a stopping criterion.  With emcee, checking for convergence is highly non-trivial.
   
* **My corner plots look ugly.  What do I do?**
  
  If you're using nested sampling, increase the number of live points. This
  will increase the number of samples your corner plot is generated from: ::

    # By default, npoints is 100
    result = retriever.run_multinest(bins, depths, errors, fit_info, npoints=1000)
    
  If you're using MCMC, increase nsteps from the default of 1000 to 10,000.

* **How do I get statistics from the retrieval?**

  Look at BestFit.txt.  It'll have the 16th, 50th, and 84th percentiles of
  all parameters, as well as the best fit values.
    
* **How do I retrieve individual species abundances?**
  You can't.  While this would be trivial to implement--and you can do so if
  you really need to--it could easily lead to combinations of species
  that are unstable on very short timescales.  We have therefore decided not
  to support retrieving on individual abundances.
  
* **PLATON is still too slow!  How do I make it faster?**

  If you didn't follow the installation instructions, go back and re-read them.
  Make sure you have OpenBLAS, MKL, or another basic linear algebra library
  (BLAS) installed
  and linked to numpy.

  If PLATON is still too slow, try decreasing num_profile_heights in
  transit_depth_calculator.py (for transit depths) or
  TP_profile (for eclipse depths).  Of course, this comes at the expense of
  accuracy.  You can also delete some of the files in data/Absorption that
  correspond to molecules which contribute negligible opacity.  This has the
  effect of setting their absorption cross section to 0.
  
  In some cases, nested sampling becomes extremely inefficient with the default
  sampling method.  In those cases, pass sample="rwalk" to run_multinest, which
  will cap the sampling efficiency at 1/25, 25 being the number of random walks to take.  According to the dynesty documentation, 25 should be sufficient
  at low dimensionality (<=10), but 50 might be necessary at
  moderate dimensionality (10-20).  To change the number of random walks to 50, pass walk=50.

* **How small can I set my wavelength bins?**
  The error in the opacity sampling calculation for a given reasonably small bin is equal to the standard deviation of the
  transit/eclipse depths in that bin divided by sqrt(N), where N is the number of points in the bin.
  With the default opacity resolution of R=1000, N = 1000 * (ln(max_wavelength/min_wavelength)).  We recommend that you
  keep N above 10 to avoid unreasonably large errors.  PLATON will throw a
  warning for N <= 5.

* **What opacity resolution should I use?  How many live points**
  This is a tradeoff between running time and accuracy.  Roughly speaking,
  the running time is proportional to the resolution and to the number of live
  points.

  We recommend a staged approach to retrievals.  Exploratory data analysis can be done with R=1000 opacities and 200 live points.  In the process, intermittent spot checks should be performed with R=10,000 opacities and 200 live points to check the effect of resolution, and with R=1000 opacities and 1000 live points to check the effect of sparse sampling.  When one is satisfied with the exploratory data analysis and is ready to finalize the results, one should run a final retrieval with R=10,000 opacities and 1000 live points.  This is the approach we followed for HD 189733b, although had we stuck with the low-resolution, sparsely sampled retrieval, our posteriors would have been slightly broader, but none of our conclusions would have changed.
