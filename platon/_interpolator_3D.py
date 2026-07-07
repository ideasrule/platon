from functools import partial

import jax.numpy as jnp
import numpy as np


def fractional_index(target_xs, xs):
    """Locate target_xs in the sorted 1-D grid xs: returns (idx, frac) with
    idx in [0, len(xs)-2] and frac clamped to [0, 1] (so out-of-range targets
    take the endpoint values, matching jnp.interp).  Computed with a one-shot
    vectorized comparison instead of jnp.searchsorted, whose scan-based binary
    search lowers to a multi-kernel while loop."""
    n = xs.shape[0]
    idx = jnp.sum(target_xs[..., None] >= xs, axis=-1) - 1
    idx = jnp.clip(idx, 0, n - 2).astype(jnp.int32)
    x0 = xs[idx]
    frac = jnp.clip((target_xs - x0) / (xs[idx + 1] - x0), 0.0, 1.0)
    return idx, frac


def interp1d(target_xs, xs, data):
    """Linearly interpolate `data` (whose first axis corresponds to the sorted
    coordinates `xs`) at target_xs.  JAX-traceable."""
    isscalar = jnp.ndim(target_xs) == 0
    target_xs = jnp.atleast_1d(target_xs)
    assert data.shape[0] == xs.shape[0]

    x_lower, x_frac = fractional_index(target_xs, xs)

    x_frac = x_frac.reshape(x_frac.shape + (1,) * (data.ndim - 1))
    result = data[x_lower] * (1 - x_frac) + data[x_lower + 1] * x_frac

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

    if xp is jnp:
        x_lower, x_frac = fractional_index(target_xs, xs)
        y_lower, y_frac = fractional_index(target_ys, ys)
        x_upper = x_lower + 1
        y_upper = y_lower + 1
    else:
        x_indices = np.interp(target_xs, xs, np.arange(len(xs)))
        y_indices = np.interp(target_ys, ys, np.arange(len(ys)))

        x_lower = np.floor(x_indices).astype(int)
        x_upper = np.ceil(x_indices).astype(int)
        x_frac = x_indices - x_lower

        y_lower = np.floor(y_indices).astype(int)
        y_upper = np.ceil(y_indices).astype(int)
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


def uniform_log_lookup(x, table_x, table_y, left, right):
    """Linear interpolation of (table_x, table_y) at x, where table_x is
    uniform in log10 (np.logspace): the bracketing segment is found
    analytically instead of by binary search.  The interpolation weight within
    the segment is linear in x, matching jnp.interp; `left`/`right` are the
    values returned outside the table range."""
    n = table_x.shape[0]
    log_x0 = jnp.log10(table_x[0])
    scale = (n - 1) / (jnp.log10(table_x[-1]) - log_x0)
    idx = (jnp.log10(x) - log_x0) * scale
    idx = jnp.clip(idx, 0, n - 2).astype(jnp.int32)
    x0 = table_x[idx]
    frac = (x - x0) / (table_x[idx + 1] - x0)
    y = table_y[idx] * (1 - frac) + table_y[idx + 1] * frac
    y = jnp.where(x < table_x[0], left, y)
    return jnp.where(x > table_x[-1], right, y)
