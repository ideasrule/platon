Quick start
***********

The fastest way to get started is to look at the examples/ directory, which
has examples on how to compute transit depths from planetary parameters, and
on how to retrieve planetary parameters from transit depths.  This page is
a short summary of the more detailed examples.

To compute transit depths, look at transit_depth_example.py, then go to
:class:`.TransitDepthCalculator` for more info.  In short::

  from pyexotransmit.transit_depth_calculator import TransitDepthCalculator

  star_radius = 7e8 # all quantities in SI
  planet_g = 9.8
  planet_radius = 7e7
  planet_temperature = 1200

  calculator = TransitDepthCalculator(star_radius, planet_g)
  calculator.compute_depths(planet_radius, planet_temperature)

You can adjust a variety of parameters, including the metallicity and C/O
ratio.  You can also specify custom abundances, such as by providing the
filename of one of the EOS files included in the package.

To retrieve on atmospheric parameters, look at retrieve_example.py, then go to
:class:`.Retriever` for more info.  In short::

  from pyexotransmit.fit_info import FitInfo
  from pyexotransmit.retrieve import Retriever

  # Set your best guess
  fit_info = retriever.get_default_fit_info(star_radius, planet_g, planet_radius, planet_temperature, logZ=0)

  # Decide what you want to fit for, then set the lower and upper limits for
  # those quantities
  
  fit_info.add_fit_param('R', 0.9*planet_radius, 1.1*planet_radius)
  fit_info.add_fit_param('T', 0.5*planet_temperature, 1.5*planet_temperature)
  fit_info.add_fit_param("logZ", -1, 2)

  result = retriever.run_multinest(bins, depths, errors, fit_info)

Here, `bins` is a N x 2 array representing the start and end wavelengths of the
bins, in metres; `depths` is a list of N transit depths; and `errors` is a list
of N errors on those transit depths.

The example above retrieves the planetary radius (at a base pressures
of 100,000 Pa), the temperature of the isothermal atmosphere, and the
metallicity.  Other parameters you can retrieve for are the C/O ratio,
the cloudtop pressure, the scattering factor, the scattering slope,
and the error multiple--which multiplies all errors by a constant.

Once you get the `result` object, you can make a corner plot::

  fig = corner.corner(result.samples, weights=result.weights, range=[0.99] * result.samples.shape[1], labels=fit_info.fit_param_names)

Additionally, result.logl stores the log likelihoods of the points in
result.samples.
