Mie scattering
**************

By default, PLATON uses a parametric model to account for scattering, with
an amplitude and a slope.  However, PLATON also has the ability to compute Mie
scattering in place of the parametric model.

To use Mie scattering, follow :doc:`quickstart` to see how to do forward models
and retrievals using the default parametric model.  To use Mie scattering
instead::

  calculator.compute_depths(Rs, Mp, Rp, T,
      ri = 1.33-0.1j, frac_scale_height = 0.5, number_density = 1e9,
      part_size = 1e-6, cloudtop_pressure=1e5)

This computes Mie scattering for particles with complex refractive index
1.33-0.1j.  The particles follow a lognormal size distribution with a mean
radius of 1 micron and standard deviation of 0.5.  They have a density of
:math:`10^9/m^3` at the cloud-top pressure of :math:`10^5` Pa, declining with
altitude with a scale height of 0.5 times the gas scale height.

We also allow the computation of Mie scattering for three condensates using
their actual, wavelength-dependent refractive indices, assuming a standard
deviation in the lognormal size distribution of 0.5::

  calculator.compute_depths(Rs, Mp, Rp, T,
      ri = "TiO2_anatase", frac_scale_height = 0.5, number_density = 1e9,
      part_size = 1e-6, cloudtop_pressure=1e5)

The supported species are those with `data in LX-MIE <https://github.com/NewStrangeWorlds/LX-MIE/tree/master/compilation>`, with the exception of Fe2SiO4 and MgAl2O4 (which do not have refractive index data all the way to 0.2 um).  

To retrieve Mie scattering parameters, make sure to set log_scatt_factor to 0,
and log_number_density to a finite value.  n and log_k specify the real component and log10 of the imaginary component of the complex refractive index.  We recommend fixing at least n.  Example::

  fit_info = retriever.get_default_fit_info(Rs, Mp, Rp, T,
      log_scatt_factor = 0, log_number_density = 9, n = 1.33, log_k=-1)

  fit_info.add_uniform_fit_param('log_number_density', 5, 15)
  fit_info.add_uniform_fit_param('log_part_size', -7, -4)
