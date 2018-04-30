# PyExoTransmit

PyExoTransmit calculates the transmission spectrum of a planet from 300 nm to 30 um, taking into account gas absorption, collisionally induced gas absorption, and Rayleigh scattering.  It is derived from ExoTransmit by Eliza Kempton (https://github.com/elizakempton/Exo_Transmit).

PyExoTransmit is written entirely in Python and is fast enough to be used in the inner loop of emcee.  By default, the code calculates transit depths
on a fine wavelength grid (λ/Δλ = 1000 with 4616 wavelength points), which takes ~0.4 seconds. The user can instead specify bins which are
directly relevant to matching observational data, in which case the code is much faster--on the order of 5 milliseconds for 30 bins.


# Installation

The package can be installed via pip:

```
pip install pyexotransmit
```

Alternatively, it can be installed from source via:

```
python setup.py install
```

pyexotransmit requires numpy and scipy, but otherwise has no dependencies.

# Usage

First, we initialize the transit calculator object with the planetary radius, stellar radius, and surface gravity, all numbers being in SI:

```
from transit_depth_calculator import TransitDepthCalculator
depth_calculator = TransitDepthCalculator(6.4e6, 7e8, 9.8)
```

Constructing this object is time consuming (hundreds of milliseconds) because it
reads in all the required data files.  It is best to construct the object once
and hold it, not reconstruct it over and over again.

Optionally, we specify the wavelength bins for which we want the transit depth:

```
depth_calculator.change_wavelength_bins([[100e-9, 200e-9], [200e-9, 300e-9]])

OR

depth_calculator.change_wavelength_bins([100e-9, 200e-9, 300e-9])
```

Then, we calculate the transit depth by specifying the atmospheric P/T profile and the abundance grid:

```
depth_calculator.compute_depths(P, T, abundances)
```

The P/T profile of the atmosphere and the abundance grid can we derived from any source, but one easy way is to load the provided data files:

```
import numpy as np
import eos_reader

index, P, T = np.load("T_P/t_p_800K.dat", unpack=True, skiprows=1)
abundances = eos_reader.get_abundances("EOS/eos_1Xsolar_cond.dat")   
```
