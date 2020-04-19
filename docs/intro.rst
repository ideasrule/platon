Introduction
************

PLATON (PLanetary Atmospheric Transmission for Observer Noobs) is a
fast and easy to use forward modelling and retrieval tool for
exoplanet atmospheres.  It is based on ExoTransmit by Eliza Kempton.
The two main modules are:

   1. :class:`.TransitDepthCalculator`: computes a transit spectrum for an
      exoplanet
   2. :class:`.EclipseDepthCalculator`: computes an eclipse spectrum   
   3. :class:`.Retriever`:  retrieves atmospheric properties of an exoplanet,
      given the observed transit spectrum.  The properties that can be retrieved
      are metallicity, C/O ratio, cloudtop pressure, scattering strength,
      and scattering slope
   4. :class:`.CombinedRetriever`: can retrieve atmospheric properties for
      eclipse depths, or a combination of transit and eclipse depths

The transit spectrum is calculated from 300 nm to 30 um, taking into
account gas absorption, collisionally induced gas absorption, clouds, 
and scattering.  :class:`.TransitDepthCalculator` is written
entirely in Python and is designed for performance. By default, it
calculates transit depths on a fine wavelength grid (λ/Δλ = 1000 with
4616 wavelength points), which takes ~65 milliseconds on a midrange
consumer computer.  The user can instead specify bins which are
directly relevant to matching observational data, in which case the
code avoids computing depths for irrelevant wavelengths and is many
times faster.  The user can also download higher resolution data (R=10,000 or R=375,000) from `here <http://astro.caltech.edu/~mz/absorption.html>`_
and drop them into PLATON's data folder; the runtime is roughly proportional
to the resolution.

The eclipse spectrum is calculated with the same physics included, but it does
not include scattering as a source of emission; scattering is only included as
a source of absorption.

The retrievers use TransitDepthCalculator/EclipseDepthCalculator as a forward
model, and can retrieve atmospheric properties using either MCMC or nested
sampling. Typically, nestled sampling with only transit depths finishes in < 10 min.  MCMC relies on the
user to specify the number of iterations, but typically reaches convergence in
less than an hour.  Eclipse depths typically take longer to calculate by a factor of a few, resulting in longer retrievals.
