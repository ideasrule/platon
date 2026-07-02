import jax.numpy as jnp
import numpy as np


def get_condition_array(target_data, interp_data, max_cutoff=np.inf):
    """Numpy helper retained for host-side (non-JIT) code paths."""
    target_data = np.asarray(target_data)
    interp_data = np.asarray(interp_data)
    cond = np.zeros(len(interp_data), dtype=bool)

    start_index = None
    end_index = None

    for i in range(len(cond)):
        if start_index is None:
            if interp_data[i] > target_data.min():
                start_index = max(0, i - 1)
            if interp_data[i] == target_data.min():
                start_index = i
        if end_index is None:
            if interp_data[i] >= target_data.max() or \
               interp_data[i] >= max_cutoff:
                end_index = i + 1

    cond[start_index: end_index] = True
    return cond


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


def regular_grid_interp(ys, xs, data, target_ys, target_xs):
    """Bilinear interpolation of `data` (dims (NY, NX, ...)) at the paired
    coordinates (target_ys, target_xs).  JAX-traceable."""
    isscalar = jnp.ndim(target_ys) == 0
    target_ys = jnp.atleast_1d(target_ys)
    target_xs = jnp.atleast_1d(target_xs)

    assert data.shape[0] == ys.shape[0] and data.shape[1] == xs.shape[0]

    x_indices = jnp.interp(target_xs, xs, jnp.arange(len(xs), dtype=data.dtype))
    y_indices = jnp.interp(target_ys, ys, jnp.arange(len(ys), dtype=data.dtype))

    x_lower = jnp.floor(x_indices).astype(jnp.int32)
    x_upper = jnp.ceil(x_indices).astype(jnp.int32)
    x_frac = x_indices - x_lower

    y_lower = jnp.floor(y_indices).astype(jnp.int32)
    y_upper = jnp.ceil(y_indices).astype(jnp.int32)
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


def regular_grid_interp_np(ys, xs, data, target_ys, target_xs):
    """Pure-numpy version of regular_grid_interp, for host-side code."""
    isscalar = np.isscalar(target_ys)
    target_ys = np.atleast_1d(target_ys)
    target_xs = np.atleast_1d(target_xs)

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
