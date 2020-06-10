Introduction
************

PLATON (PLanetary Atmospheric Transmission for Observer Noobs) is a
fast and easy to use forward modelling and retrieval tool for
exoplanet atmospheres.  It is based on ExoTransmit by Eliza Kempton.
The two main modules are:

   1. :class:`.TransitDepthCalculator`: computes a transit spectrum for an
      exoplanet
   2. :class:`.EclipseDepthCalculator`: computes an eclipse spectrum   
   3. :class:`.CombinedRetriever`: can retrieve atmospheric properties for
      transit depths, eclipse depths, or a combination of the two.

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
sampling.  The speed of these retrievals is highly dependent on the wavelength
range, data precision, prior ranges, opacity resolution, and number of live points (nested sampling)
or iterations/walkers (MCMC).  A very rough guideline is that a retrieval with
200 live points and R=1000 (suitable for exploratory work) for
STIS + WFC3 + IRAC 3.6 um + IRAC 4.5 um data takes <1 hour, while a
retrieval with 1000 live points and R=10,000 (suitable for the final version)
takes 1-2 days.  There are a variety of ways to speed up the retrieval, as
described in our PLATON II paper.  These include using correlated-k instead of
opacity sampling with R=10,000, or removing the opacity data files of
unimportant molecules (thereby zeroing their opacities).
