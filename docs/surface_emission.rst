Surface emission (beta)
=======================
PLATON can now calculate the emission from a rocky planet's surface, using realistic albedos newly measured in a lab by Caltech graduate student Kim Paragas.  This feature should be considered beta until Paragas et al. (in prep) comes out.

To try out this feature, take a look at examples/surface_example.py.  One begins by initializing an eclipse depth calculator with either the "HES2012" (Hu, Ehlmann, Seager 2012) spectral library or the new "Paragas" library::
  
  calc = EclipseDepthCalculator(surface_library="Paragas")

Then, call compute_depths with all the ordinary eclipse calculator arguments, but also the surface arguments::

  wavelengths, depths, info_dict = calc.compute_depths(...,
    surface_pressure = Psurf, surface_type=surface_type, semimajor_axis=semi_major_axis, surface_temp = surface_temp

)

The available surface types are the columns in data/HES2012/hemi_refls.csv or data/Paragas/hemi_refls.csv.  surface_temp can be None, in which case the temperature will be automatically calculated from energy balance.
