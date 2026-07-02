from functools import partial

import jax.numpy as jnp
import numpy as np


def interp1d(target_xs, xs, data):
    """Linearly interpolate `data` (whose first axis corresponds to the sorted
    coordinates `xs`) at target_xs.  JAX-traceable."""
    isscalar = jnp.ndim(target_xs) == 0
    target_xs = jnp.atleast_1d(target_xs)
    assert data.shape[0] == xs.shape[0]

    x_indices = jnp.interp(target_xs, xs, jnp.arange(len(xs), dtype=data.dtype))
    x_lower = jnp.floor(x_indices).astype(jnp.int32)
    x_upper = jnp.ceil(x_indices).astype(jnp.int32)
    x_frac = x_indices - x_lower

    x_frac = x_frac.reshape(x_frac.shape + (1,) * (data.ndim - 1))
    result = data[x_lower] * (1 - x_frac) + data[x_upper] * x_frac

    if isscalar:
        return result[0]
    return result


def interp1d_np(target_xs, xs, data):
    """Pure-numpy (float64) version of interp1d, for host-side code."""
    isscalar = np.isscalar(target_xs)
    target_xs = np.atleast_1d(target_xs)
    assert data.shape[0] == len(xs)

    x_indices = np.interp(target_xs, xs, np.arange(len(xs)))
    x_lower = np.floor(x_indices).astype(int)
    x_upper = np.ceil(x_indices).astype(int)
    x_frac = x_indices - x_lower

    x_frac = x_frac.reshape(x_frac.shape + (1,) * (data.ndim - 1))
    result = data[x_lower] * (1 - x_frac) + data[x_upper] * x_frac

    if isscalar:
        return result[0]
    return result


def _regular_grid_interp(ys, xs, data, target_ys, target_xs, xp):
    """Bilinear interpolation of `data` (dims (NY, NX, ...)) at the paired
    coordinates (target_ys, target_xs)."""
    isscalar = xp.ndim(target_ys) == 0
    target_ys = xp.atleast_1d(target_ys)
    target_xs = xp.atleast_1d(target_xs)

    assert data.shape[0] == len(ys) and data.shape[1] == len(xs)

    index_dtype = data.dtype if xp is jnp else np.float64
    x_indices = xp.interp(target_xs, xs, xp.arange(len(xs), dtype=index_dtype))
    y_indices = xp.interp(target_ys, ys, xp.arange(len(ys), dtype=index_dtype))

    x_lower = xp.floor(x_indices).astype(jnp.int32 if xp is jnp else int)
    x_upper = xp.ceil(x_indices).astype(jnp.int32 if xp is jnp else int)
    x_frac = x_indices - x_lower

    y_lower = xp.floor(y_indices).astype(jnp.int32 if xp is jnp else int)
    y_upper = xp.ceil(y_indices).astype(jnp.int32 if xp is jnp else int)
    y_frac = y_indices - y_lower

    extra_dims = data.ndim - 2
    x_frac = x_frac.reshape(x_frac.shape + (1,) * extra_dims)
    y_frac = y_frac.reshape(y_frac.shape + (1,) * extra_dims)

    result = data[y_lower, x_lower] * (1 - y_frac) * (1 - x_frac) + \
             data[y_upper, x_lower] * y_frac * (1 - x_frac) + \
             data[y_lower, x_upper] * (1 - y_frac) * x_frac + \
             data[y_upper, x_upper] * y_frac * x_frac
    if isscalar:
        return result[0]
    return result


regular_grid_interp = partial(_regular_grid_interp, xp=jnp)     # JAX-traceable
regular_grid_interp_np = partial(_regular_grid_interp, xp=np)   # host float64
