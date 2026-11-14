Quick start
***********

The fastest way to get started is to look at the examples/ directory, which
has examples on how to compute transit/eclipse depths from planetary parameters, and
on how to retrieve planetary parameters from transit/eclipse depths.  This page is
a short summary of the more detailed examples.

To compute transit depths, look at transit_depth_example.py, then go to
:class:`.TransitDepthCalculator` for more info.  In short::

  from platon.transit_depth_calculator import TransitDepthCalculator
  from platon.TP_profile import Profile
  from platon.constants import M_jup, R_jup, R_sun

  # All inputs and outputs for PLATON are in SI
  
  Rs = 1.16 * R_sun
  Mp = 0.73 * M_jup
  Rp = 1.40 * R_jup
  T = 1200

  # The initializer loads all data files.  Create a TransitDepthCalculator
  # object and hold on to it
  calculator = TransitDepthCalculator()

  # Can also use Guillot 2010, Line et al 2013, Madhusudhan & Seager 2009, or custom profiles
  p = Profile.isothermal(T)
  
  # compute_depths is fast once data files are loaded
  wavelengths, depths, info_dict = calculator.compute_depths(p, Rs, Mp, Rp, logZ=0, CO_ratio=0.5, full_output=True)

You can adjust a variety of parameters, including the metallicity (Z) and C/O
ratio.  full_output=True indicates you'd like extra information about the atmosphere, which is returned in info_dict.  info_dict includes parameters like the
temperatures, pressures, radii, abundances, and molecular weights of each
atmospheric layer, and the line of sight optical depth (tau_los) through each
layer.  It also includes the contribution function.

You can also specify custom abundance profiles by setting logZ and CO_ratio
to None and passing custom_abundances.  See plot_transit_custom_abunds.py for an example.

To retrieve atmospheric parameters, look at retrieve_k2_18b_transit.py or retrieve_hd189_eclipse.py, then go to
:class:`.CombinedRetriever` for more info.  The former performs a free retrieval, while the latter assumes equilibrium chemistry.  

The example adopts uniform log VMRs, but it is also possible to use CLR priors::

  fit_info = retriever.get_default_fit_info(..., logZ=None, CO_ratio=None, fit_vmr=False, fit_clr=True)
  fit_info.add_gases_clr(["CH4", "CO2", "CO", "H2O", "H2S"])

With a CLR prior, all gases are treated equally, and there is no pre-defined background gas.

In an equilibrium retrieval, you can still retrieve the abundances of individual gases, such as
photochemical products that equilibrium chemistry underpredicts.  Fit for log_<gas>, the log10 of
the volume mixing ratio::

  fit_info = retriever.get_default_fit_info(..., logZ=0, CO_ratio=0.53)
  fit_info.add_uniform_fit_param("logZ", -1, 3)
  fit_info.add_uniform_fit_param("log_CS2", -12, -2)
  fit_info.add_uniform_fit_param("log_CH4", -12, -2)

This sets the CS2 and CH4 abundances to 10**log_CS2 and 10**log_CH4 at all pressures, while all other
species remain in chemical equilibrium.  The other abundances are not renormalized, so this is intended
for trace gases.  To fix an abundance instead of fitting it, pass it to get_default_fit_info (e.g. log_SO2=-5.5).
The forward models accept the same thing through the gases and vmrs arguments of compute_depths:
``calculator.compute_depths(p, Rs, Mp, Rp, logZ=0, CO_ratio=0.53, gases=["CS2"], vmrs=[1e-6])``.

The examples use the nautilus package for importance nested sampling.  You can also use dynesty, multinest, or emcee
by changing retriever.run_nautilus to retriever.run_dynesty, retriever.run_multinest, or retriever.run_emcee and
setting the hyperparameters accordingly.  We recommend sticking with nautilus because we find that it is more robust
and sample efficient than multinest and dynesty, especially at higher dimensions.

After a retrieval, PLATON automatically computes the Pareto importance sampling leave-one-out cross-validation score (PSIS-LOO), first used in an exoplanet context by `Welbanks et al 2023 <https://ui.adsabs.harvard.edu/abs/2023AJ....165..112W/abstract>`_.  The values are stored in the retrieval pickle file as "loos" (also see "loo_total" and "loo_ks").  
