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

To retrieve Mie scattering parameters, make sure to set log_scatt_factor to 0, log_number_density to a finite value, and ri to the complex refractive
index of the scattering particles (which cannot be a free parameter)::

  fit_info = retriever.get_default_fit_info(Rs, Mp, Rp, T,
      log_scatt_factor = -np.inf, log_number_density = 9, ri = 1.33-0.1j)

  fit_info.add_uniform_fit_param('log_number_density', 5, 15)
  fit_info.add_uniform_fit_param('log_part_size', -7, -4)
