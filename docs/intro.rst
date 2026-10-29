Introduction
************

PLATON (PLanetary Atmospheric Tool for Observer Noobs) is a
fast and easy to use forward modelling and retrieval tool for
exoplanet atmospheres.  Its roots are in ExoTransmit by Eliza Kempton, but
it has dramatically changed since then.  The latest version, PLATON 7, is
built upon JAX and about a hundred times faster than PLATON 6, achieving a
speed of 0.4 ms per forward model.

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
sampling.  By default we use `nautilus <https://nautilus-sampler.readthedocs.io/en/latest/>`_, which does importance sampling using neural networks for efficient parameter space exploration.  We find that it is more robust and more sample
efficient for high-dimensional problems.
