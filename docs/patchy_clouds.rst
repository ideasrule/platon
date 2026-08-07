Patchy clouds
*************

A terminator need not be uniformly cloudy.  One part may be hidden by an
opaque cloud deck while another part remains clear.  This can mute molecular
features without requiring the entire atmosphere to have a high cloud.

PLATON follows the simple area-weighted model used by
`Line & Parmentier (2016)
<https://doi.org/10.3847/0004-637X/820/1/78>`_:

.. math::

   D(\lambda) = f_\mathrm{cloud}D_\mathrm{cloudy}(\lambda)
              + (1-f_\mathrm{cloud})D_\mathrm{clear}(\lambda).

Here ``cloud_fraction`` is the projected fraction of the terminator covered
by clouds.  It is zero for a clear terminator and one for a uniformly cloudy
terminator.

Computing a spectrum
====================

Patchiness is an option on the ordinary transmission calculation.  This
example covers 70% of the terminator with a gray cloud deck at 0.1 bar::

  from platon.constants import M_jup, R_jup, R_sun
  from platon.TP_profile import Profile
  from platon.transit_depth_calculator import TransitDepthCalculator

  profile = Profile()
  profile.set_isothermal(1200)

  calculator = TransitDepthCalculator()
  wavelengths, depths, info = calculator.compute_depths(
      profile,
      star_radius=1.19 * R_sun,
      planet_mass=0.73 * M_jup,
      planet_radius=1.40 * R_jup,
      logZ=0,
      CO_ratio=0.53,
      cloudtop_pressure=1e4,
      cloud_fraction=0.7,
      full_output=True)

The cloudy and clear calculations have the same temperature, composition,
planet radius, and stellar correction.  The clear calculation removes the
gray cloud deck, parametric haze, and Mie particles.  Ordinary molecular
Rayleigh scattering remains.

The complete synthetic retrieval in
:download:`retrieve_partial_clouds.py
<../examples/retrieve_partial_clouds.py>` uses a non-isothermal terminator and
JWST/PRISM-like wavelength bins.

Retrieving the cloud fraction
=============================

Set the starting value with
:func:`.CombinedRetriever.get_default_fit_info`.  If no prior is added, the
fraction remains fixed::

  from platon.combined_retriever import CombinedRetriever

  retriever = CombinedRetriever()
  fit_info = retriever.get_default_fit_info(
      Rs=1.19 * R_sun,
      Mp=0.73 * M_jup,
      Rp=1.40 * R_jup,
      T=1200,
      logZ=0,
      CO_ratio=0.53,
      log_cloudtop_P=4,
      cloud_fraction=0.5)

To let the data determine the coverage, add a uniform projected-area prior::

  fit_info.add_uniform_fit_param("cloud_fraction", 0, 1)

Cloud and haze properties are fitted in the usual way::

  fit_info.add_uniform_fit_param("log_cloudtop_P", -0.99, 7)
  fit_info.add_uniform_fit_param("log_scatt_factor", -2, 6)
  fit_info.add_uniform_fit_param("scatt_slope", 0, 12)

``log_cloudtop_P`` is log10 pressure in pascals.

Limitations
===========

This model contains one cloudy and one clear column with a shared thermal
profile and chemistry.  Use :doc:`two_sector_terminators` when the cold and
hot limbs need different temperatures or cloud properties.  The two models
cannot be nested: a :class:`.TwoSectorTerminator` requires
``cloud_fraction=1`` and uses ``cold_fraction`` for its own area weighting.
