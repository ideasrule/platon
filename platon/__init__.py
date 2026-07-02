name = "platon"
__version__ = "6.3.1"
__md5sum__ = "4deb845a63909ed8d0a74de2af5f0a48"
__data_url__ = "https://astro.uchicago.edu/~mz/data_{}.zip".format(__md5sum__)

# PLATON performs all heavy computation with JAX in single precision (FP32).
# jax_enable_x64 must be False so that computation actually happens in FP32.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", False)
# Do not let matmuls/einsums silently drop to TF32 (10-bit mantissa) on
# Ampere+ GPUs; full FP32 precision is required for accurate depths.
_jax_config.update("jax_default_matmul_precision", "highest")
