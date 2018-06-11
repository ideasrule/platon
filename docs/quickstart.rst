Quick start
***********

The fastest way to get started is to look at the examples/ directory, which
has examples on how to compute transit depths from planetary parameters, and
on how to retrieve planetary parameters from transit depths.  This page is
a short summary of the more detailed examples.

To compute transit depths, look at transit_depth_example.py, then go to
:class:`.TransitDepthCalculator` for more info.  In short::

  from platon.transit_depth_calculator import TransitDepthCalculator
  from platon.constants import M_jup, R_jup, R_sun

  # All inputs and outputs for PLATON are in SI
  
  Rs = 1.16 * R_sun
  Mp = 0.73 * M_jup
  Rp = 1.40 * R_jup
  T = 1200

  # The initializer loads all data files.  Create a TransitDepthCalculator
  # object and hold on to it
  calculator = TransitDepthCalculator()

  # compute_depths is fast once data files are loaded
  calculator.compute_depths(Rs, Mp, Rp, T, logZ=0, CO_ratio=0.53)

You can adjust a variety of parameters, including the metallicity (Z) and C/O
ratio. By default, logZ = 0 and C/O = 0.53. Any other value for
logZ and C/O in the range -1 < logZ < 3 and 0.2 < C/O < 2 can also be used.
You can use a dictionary of numpy arrays to specify abundances as well
(See the API).
You can also specify custom abundances, such as by providing the filename or
one of the abundance files included in the package (from ExoTransmit). The
custom abundance files specified by the user must be compatible with the
ExoTransmit format::

  calculator.compute_depths(Rs, Mp, Rp, T, logZ=None, CO_ratio=None,
                            custom_abundances=filename)

To retrieve atmospheric parameters, look at retrieve_example.py, then go to
:class:`.Retriever` for more info.  In short::

  from platon.fit_info import FitInfo
  from platon.retriever import Retriever

  # Set your best guess
  fit_info = retriever.get_default_fit_info(Rs, Mp, Rp, T, logZ=0)

  # Decide what you want to fit for, then set the lower and upper limits for
  # those quantities

  fit_info.add_fit_param('R', 0.9*planet_radius, 1.1*planet_radius)
  fit_info.add_fit_param('T', 0.5*planet_temperature, 1.5*planet_temperature)
  fit_info.add_fit_param("logZ", -1, 2)

  #Fit using Nested Sampling
  result = retriever.run_multinest(bins, depths, errors, fit_info)

Here, `bins` is a N x 2 array representing the start and end wavelengths of the
bins, in metres; `depths` is a list of N transit depths; and `errors` is a list
of N errors on those transit depths.

The example above retrieves the planetary radius (at a base pressures
of 100,000 Pa), the temperature of the isothermal atmosphere, and the
metallicity.  Other parameters you can retrieve for are the stellar radius,
the planetary mass, C/O ratio,
the cloudtop pressure, the scattering factor, the scattering slope,
and the error multiple--which multiplies all errors by a constant.  We recommend
either fixing the stellar radius and planetary mass to the measured values, or
only allowing them to vary 2 standard deviations aw

Once you get the `result` object, you can make a corner plot::

  fig = corner.corner(result.samples, weights=result.weights,
                      range=[0.99] * result.samples.shape[1],
                      labels=fit_info.fit_param_names)

Additionally, result.logl stores the log likelihoods of the points in
result.samples.

If you prefer using MCMC instead of Nested Sampling in your retrieval, you can
use the run_emcee method instead of the run_multinest method. Do note that
Nested Sampling tends to be much faster and it does not require specification
of a termination point::

  result = retriever.run_emcee(bins, depths, errors, fit_info)

For MCMC, the number of walkers and iterations/steps can also be specified. The
`result` object returned by run_emcee is slighly different from that returned
by run_multinest. To make a corner plot with the result of run_emcee::

  fig = corner.corner(result.flatchain, range=[0.99] * result.flatchain.shape[1],
                      labels=fit_info.fit_param_names)
