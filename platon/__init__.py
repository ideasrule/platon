name = "platon"
__version__ = "6.3.1"
__md5sum__ = "4deb845a63909ed8d0a74de2af5f0a48"
__data_url__ = "https://astro.uchicago.edu/~mz/data_{}.zip".format(__md5sum__)

# Capture small GPU kernel sequences into CUDA graphs (XLA's default
# threshold is conservative); saves ~15% wall time per forward model by
# eliminating kernel launch overhead.  Appended so user-set XLA_FLAGS win.
import os as _os
if "--xla_gpu_graph_min_graph_size" not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") +
                                " --xla_gpu_graph_min_graph_size=2").strip()

# PLATON performs all heavy computation with JAX in single precision (FP32).
# jax_enable_x64 must be False so that computation actually happens in FP32.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", False)
# Do not let matmuls/einsums silently drop to TF32 (10-bit mantissa) on
# Ampere+ GPUs; full FP32 precision is required for accurate depths.
_jax_config.update("jax_default_matmul_precision", "highest")
