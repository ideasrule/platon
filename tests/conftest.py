"""Keeps the memory use of the test suite bounded.

PLATON's caches share opacity data between calculators, which is what a
retrieval wants.  The test suite, though, builds calculators with many
different opacity lists, stellar grids and wavelength bins in one process,
and each combination pins its own multi-GB copy.  Without a GPU those copies
are all in host memory, which exceeds the 16 GB of a GitHub runner.

For this to work, tests must not keep calculators alive past their own
class: a calculator that outlives a cache clear forces the next test to load
a second copy of the opacities.  Build shared calculators in setUpClass (not
__init__, which pytest runs at collection) and drop them in tearDownClass.
"""
import gc

import jax
import pytest

from platon import _atmosphere_solver


@pytest.fixture(autouse=True)
def clear_device_cache():
    """After each test, free the device copies of the opacities, which are
    cheap to rebuild from the raw data."""
    yield
    _atmosphere_solver._DEVICE_CACHE.clear()
    gc.collect()


@pytest.fixture(autouse=True, scope="module")
def clear_module_caches():
    """After each test file, also free the raw opacities (slow to reload
    from disk) and JAX's compiled functions."""
    yield
    _atmosphere_solver._RAW_CACHE.clear()
    _atmosphere_solver._DEVICE_CACHE.clear()
    jax.clear_caches()
    gc.collect()
