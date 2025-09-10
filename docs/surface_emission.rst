Surface emission
=======================
PLATON can now calculate the emission from a rocky planet's surface, using realistic albedos newly measured in a lab by Caltech graduate student Kim Paragas (see Paragas et al. 2025 for more details).

To try out this feature, take a look at examples/surface_example.py. One begins by initializing an eclipse depth calculator with either the "HES2012" (Hu, Ehlmann, Seager 2012) spectral library or the new "Paragas" library::

  calc = EclipseDepthCalculator(surface_library="Paragas")

Optionally, one can estimate the surface temperature of a chosen surface_type from energy balance before computing the secondary eclipse depths by first retrieving the appropriate stellar spectrum and then calling calc_surface_temp::

    stellar_fluxes_orig, _ = calc.atm.get_stellar_spectrum(calc.atm.orig_lambda_grid, T_star, stellar_blackbody=False)
    surface_temp = calc.calc_surface_temp(surface_type, stellar_fluxes_orig, semi_major_axis / star_radius)

Then, call compute_depths with all the ordinary eclipse calculator arguments, but also the surface arguments::

  wavelengths, depths, info_dict = calc.compute_depths(...,
    surface_pressure = Psurf, surface_type=surface_type, semimajor_axis=semi_major_axis, surface_temp = surface_temp)

surface_temp can be None, in which case the temperature will be automatically calculated from energy balance.
The available surface types are the columns in data/HES2012/hemi_refls.csv or data/Paragas/hemi_refls.csv. The "Paragas" library surface types include a range of textures.

One can also see the surface temperature after the run::

    surface_temp = info_dict['surface_temperature']

A commonly asked question is **how do I generate the emission spectrum of an airless planet?**  We have yet to implement a simple way to do this, but it's easy enough to hack.
In surface_example.py, set Psurf to a negligible value, set the atmospheric temperature to the surface temperature, and give the atmosphere a composition that has no features::

  Psurf = 2e-4
  ...
  abundances = AbundanceGetter().get(0, 0.5)
  for key in abundances:
    abundances[key] *= 0
  abundances["N2"] += 1
  ...
  p.set_isothermal(surface_temp)

This is way overkill because the first and third steps alone should be enough to simulate an airless body, but better safe than sorry. 
