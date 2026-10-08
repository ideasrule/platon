1.5-D transmission spectra
****************************

A 1D transmission model assumes that the whole terminator has the same
temperature and cloud properties.  This is often a useful approximation, but
the morning and evening sides of a planet need not be alike.  PLATON can model
the terminator as one cold sector and one hot sector.

The two sectors are ordinary 1D atmospheres.  Their transit depths are combined
according to the fraction of the terminator occupied by the cold sector:

.. math::

   D(\lambda) = f_\mathrm{cold}D_\mathrm{cold}(\lambda)
              + (1-f_\mathrm{cold})D_\mathrm{hot}(\lambda).

This is sometimes called a 2D retrieval.  Here we call it 1.5-D because it
does not calculate horizontal transport or trace rays through a continuous 2D
atmosphere.

Computing a spectrum
====================

First make a temperature profile and a :class:`.TerminatorSector` for each
side.  The example below gives the cold sector a high cloud and a stronger
haze::

  from platon.constants import AU, M_jup, R_jup, R_sun
  from platon.terminator import TerminatorSector, TwoSectorTerminator
  from platon.TP_profile import Profile
  from platon.transit_depth_calculator import TransitDepthCalculator

  Rs = 1.16 * R_sun
  Mp = 0.73 * M_jup
  Rp = 1.40 * R_jup

  cold_profile = Profile.isothermal(900)
  cold = TerminatorSector(
      cold_profile,
      cloudtop_pressure=1e3,
      scattering_factor=100,
      scattering_slope=6)

  hot_profile = Profile.isothermal(1400)
  hot = TerminatorSector(
      hot_profile,
      cloudtop_pressure=1e6,
      scattering_factor=1,
      scattering_slope=4)

  terminator = TwoSectorTerminator(cold, hot, cold_fraction=0.5)
  calculator = TransitDepthCalculator()
  wavelengths, depths, info = calculator.compute_depths(
      terminator, Rs, Mp, Rp,
      logZ=0, CO_ratio=0.53, full_output=True)

``cold_fraction=0.5`` gives both sectors the same projected area.  The two
individual calculations are available as ``info["sectors"]["cold"]`` and
``info["sectors"]["hot"]``.  Each contains its own temperatures, radii,
abundances, and optical depths.

:download:`Download the complete spectrum example
<../examples/two_sector_transit.py>`.

Using Guillot profiles
======================

The same interface can use a Guillot profile instead of an isothermal one.
As in the Line et al. (2013) profile, the irradiation temperature is
:math:`\beta T_\star \sqrt{R_\star/2a}`, where ``beta`` absorbs the albedo
and heat redistribution.  The thermal opacity ``log_k_th`` is in
log10(m2/kg).  The star, orbit, planet, ``log_k_th``, and ``T_int`` are
shared by the two sectors, while ``beta`` and ``log_gamma`` may differ::

  T_star = 6100
  a = 0.047 * AU

  cold_profile = Profile.guillot(
      T_star=T_star, Rs=Rs, a=a, Mp=Mp, Rp=Rp,
      beta=0.82, log_k_th=-3, log_gamma=-1.2, T_int=150)

  hot_profile = Profile.guillot(
      T_star=T_star, Rs=Rs, a=a, Mp=Mp, Rp=Rp,
      beta=1.16, log_k_th=-3, log_gamma=-0.6, T_int=150)

  terminator = TwoSectorTerminator(
      TerminatorSector(cold_profile, cloudtop_pressure=1e3),
      TerminatorSector(hot_profile, cloudtop_pressure=1e6),
      cold_fraction=0.5)

Cold and hot refer to the mean temperature between 0.1 mbar and 1 bar,
roughly where transmission spectra form.  Guillot profiles can cross when
their ``log_gamma`` values differ, so the sector with the larger ``beta`` is
not necessarily the hot one.

Bonus: quench pressure
======================

Vertical mixing can carry gas upward faster than chemical reactions can
restore equilibrium.  A compact way to represent this is to choose a quench
pressure.  Above that pressure, PLATON holds every equilibrium species at its
abundance at the quench point.

The two Guillot sectors share one pressure, but each sector has its own
temperature there.  The cold chemistry is therefore frozen at
``cold_profile(P_quench)`` and the hot chemistry at
``hot_profile(P_quench)``.  Pass the pressure in pascals when computing a
spectrum::

  P_quench = 1e5  # 1 bar
  wavelengths, depths, info = calculator.compute_depths(
      terminator, Rs, Mp, Rp,
      logZ=0, CO_ratio=0.53,
      P_quench=P_quench,
      full_output=True)

For a retrieval, ``log_P_quench`` is log10 pressure in pascals.  The following
prior spans the full pressure grid used by the standard PLATON profile::

  fit_info.add_uniform_fit_param("log_P_quench", -4, 8)

`Taylor et al. (2026)
<https://arxiv.org/abs/2607.06491>`_ found that fitting quench pressures can
avoid biased bulk composition estimates when vertical mixing is present.
Their parameterization used separate pressures for carbon- and
nitrogen-bearing species.  PLATON's option is deliberately simpler: one
shared pressure freezes all equilibrium species.

Retrieving two sectors
======================

Pass the terminator to
:func:`.CombinedRetriever.get_default_fit_info`.  Its sectors are fitted
under the neutral names ``sector1`` and ``sector2``, each with its own
independent priors.  Afterwards, each sample's colder sector is labelled
``cold``, with all of its parameters, so the corner plot and ``BestFit.txt``
show ``cold.T``, ``hot.T``, ``cold.log_cloudtop_P``, ... and
``cold_fraction``::

  from platon.combined_retriever import CombinedRetriever

  retriever = CombinedRetriever()
  fit_info = retriever.get_default_fit_info(
      Rs=Rs, Mp=Mp, Rp=Rp, T=None,
      logZ=0, CO_ratio=0.53, T_star=6100,
      transit_terminator=terminator)

  for sector in ("sector1", "sector2"):
      fit_info.add_uniform_fit_param(sector + ".T", 300, 3000)
      fit_info.add_uniform_fit_param(sector + ".log_cloudtop_P", -0.99, 7)
      fit_info.add_uniform_fit_param(sector + ".log_scatt_factor", -2, 6)
      fit_info.add_uniform_fit_param(sector + ".scatt_slope", 0, 12)

  result = retriever.run_dynesty(
      bins, depths, errors,
      None, None, None,
      fit_info)

In this example the fraction stays fixed at its initial value of 0.5.  To fit
it freely, add one more uniform prior before running the retrieval::

  fit_info.add_uniform_fit_param("sector1.fraction", 0, 1)

The fraction prior is uniform in projected area.  Because the sectors are
unordered, the raw posterior has two mirror-image modes, which the labelling
combines.  Give both sectors the same priors.

For a Guillot retrieval, fit ``sector1.beta`` and ``sector2.beta`` instead
of the temperatures.  The shared ``T_star``, ``Rs``, ``a``,
``Mp``, and ``Rp`` are taken from the terminator; any of these also passed
to ``get_default_fit_info`` must agree with it.  The other profile
parameters are ordinary uniform fit parameters::

  for sector in ("sector1", "sector2"):
      fit_info.add_uniform_fit_param(sector + ".beta", 0.4, 1.8)
      fit_info.add_uniform_fit_param(sector + ".log_gamma", -3, 1)
  fit_info.add_uniform_fit_param("log_k_th", -5, 0)

The standard spectrum and corner plots work with a 1.5-D result.  The
temperature-profile plot draws the cold and hot posterior regions separately::

  from platon.plotter import Plotter

  Plotter.plot_retrieval_transit_spectrum(result, prefix="two_sector")
  Plotter.plot_retrieval_corner(
      result, filename="two_sector_corner.png")
  Plotter.plot_retrieval_TP_profiles(result, prefix="two_sector")

Interpreting the fraction
=========================

The limb fraction and temperature contrast can be strongly degenerate.  For
example, a small hot sector with a large contrast may resemble two equal
sectors with a smaller contrast.  They are not generally identical because
scale height, equilibrium chemistry, clouds, and Guillot profiles all change
the spectrum nonlinearly.

If the data do not constrain the fraction, leaving it fixed at 0.5 often makes
the retrieval easier to interpret.  Fitting it with a uniform prior is useful
when the wavelength coverage and precision can distinguish the sector
spectra.  PLATON samples the free fraction explicitly; it does not marginalize
over it analytically.

Limitations
===========

The two sectors share their bulk composition and quench pressure, although
equilibrium abundances are evaluated on each T/P profile.  Each sector is
homogeneous, so the existing ``cloud_fraction`` option must remain one.  This
model is a compact description of an asymmetric terminator, not a circulation
or horizontally coupled atmosphere model.
