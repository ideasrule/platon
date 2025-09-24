Introduction
************

PLATON (PLanetary Atmospheric Tool for Observer Noobs) is a
fast and easy to use forward modelling and retrieval tool for
exoplanet atmospheres.  Its roots are in ExoTransmit by Eliza Kempton, but
it has dramatically changed since then.

The main modules are:

   1. :class:`.TransitDepthCalculator`: computes a transit spectrum for an
      exoplanet
   2. :class:`.EclipseDepthCalculator`: computes an eclipse spectrum   
   3. :class:`.CombinedRetriever`: can retrieve atmospheric properties for
      transit depths, eclipse depths, or a combination of the two.

The transit spectrum is calculated from 0.2 um to 30 um, taking into
account gas absorption, H- absorption, collisionally induced absorption,
clouds, and scattering.  The eclipse spectrum is calculated with the same physics included, but it does not include scattering as a source of emission; scattering is only included as a source of absorption.

The retrievers use TransitDepthCalculator/EclipseDepthCalculator as a forward
model, and can retrieve atmospheric properties using either MCMC or nested
sampling.  The speed of these retrievals is highly dependent on the wavelength
range, data precision, prior ranges, opacity resolution, and number of live points (nested sampling)
or iterations/walkers (MCMC).  A very rough guideline is that a GPU-accelerated transit retrieval with
250 live points and R=20k for
STIS + WFC3 + IRAC 3.6 um + IRAC 4.5 um data takes 19 minutes with multinest, or 75 minutes with dynesty.  There are a variety of ways to speed up the retrieval, as
described on the Q&A page. 
