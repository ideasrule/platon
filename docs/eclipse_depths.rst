Eclipse depths
=====================

Although PLATON began life as a transmission spectrum calculator, we have also
written an eclipse depth calculator and retriever.

To use the eclipse depth calculator, first create a temperature-pressure
profile::

  from platon.TP_profile import Profile
  p = Profile()
  p.set_parametric(1200, 500, 0.5, 0.6, 1e6, 1900)

This creates a parametric T-P profile according to `Madhusudhan & Seager 2009 <https://arxiv.org/pdf/0910.1347.pdf>`_.  The parameters are: T\ :sub:`0`\, P\ :sub:`1`\, α\ :sub:`1`\, α\ :sub:`2`\, P\ :sub:`3`\, T\ :sub:`3`\.  P\ :sub:`0` \ is set to 10\ :sup:`-4` \ Pa, while P\ :sub:`2` \ and T\ :sub:`2` \ are derived from the six specified parameters.

Then, call the eclipse depth calculator::

  from eclipse_depth_calculator import EclipseDepthCalculator
  calc = EclipseDepthCalculator(method="xsec") #"ktables" for correlated k
  wavelengths, depths = calc.compute_depths(p, Rs, Mp, Rp, Tstar)
  
Most of the same parameters accepted by the transit depth calculator are also
accepted by the eclipse depth calculator.

It is also possible to retrieve on combined transit and eclipse depths::

  from platon.combined_retriever import CombinedRetriever

  retriever = CombinedRetriever()
  fit_info = retriever.get_default_fit_info(Rs, Mp, Rp, T_limb,
                 T0=1200, P1=500, alpha1=0.5, alpha2=0.6, P3=1e6, T3=1900)
		 
  fit_info.add_uniform_fit_param(...)
  fit_info.add_uniform_fit_param(...)

  result = retriever.run_multinest(transit_bins, transit_depths, transit_errors,
                                   eclipse_bins, eclipse_depths, eclipse_errors,
				   fit_info,
				   rad_method="xsec") #"ktables" for corr-k

Here, T_limb is the temperature at the planetary limb (used for transit depths),
while the T-P profile parameters are for the dayside (used for eclipse depths).

  

