Introduction
************

PLATON (PLanetary Atmospheric Transmission for Observer Noobs) is a
fast and easy to use forward modelling and retrieval tool for
exoplanet atmospheres.  It is based on ExoTransmit by Eliza Kempton.
The two main modules are:

   1. :class:`.TransitDepthCalculator`: computes a transit spectrum for an
      exoplanet
   2. :class:`.Retriever`:  retrieves atmospheric properties of an exoplanet,
      given the observed transit spectrum.  The properties that can be retrieved
      are metallicity, C/O ratio, cloudtop pressure, scattering strength,
      and scattering slope

The transit spectrum is calculating from 300 nm to 30 um, taking into
account gas absorption, collisionally induced gas absorption, and
Rayleigh scattering.  :class:`.TransitDepthCalculator` is written
entirely in Python and is designed for performance. By default, it
calculates transit depths on a fine wavelength grid (λ/Δλ = 1000 with
4616 wavelength points), which takes ~170 milliseconds on a midrange
consumer computer.  The user can instead specify bins which are
directly relevant to matching observational data, in which case the
code avoids computing depths for irrelevant wavelengths and is many
times faster.

:class:`.Retriever` uses TransitDepthCalculator as a forward model, and
can retrieve atmospheric properties using either MCMC or nested sampling.
Typically, nestled sampling finishes in < 10 min.  MCMC relies on the user to
specify the number of iterations, but typically reaches convergence in less
than an hour.
