Quick start
***********

The fastest way to get started is to look at the examples/ directory, which
has examples on how to compute transit depths from planetary parameters, and
on how to retrieve planetary parameters from transit depths.  This page is
a short summary of the more detailed examples.

To compute transit depths::

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

