"""JAX kernels for the hot path of Mie scattering with a log-normal size
distribution (AtmosphereSolver.get_mie_eff_cross_section).

Two kernels, both jitted and run on JAX's default device (the GPU, if any):

* ``spline_lognormal_integral`` evaluates the not-a-knot cubic spline of
  Qext(log x) at every (wavelength, radius) pair and takes the weighted sum
  over radii, in float32.  The float32 subtlety is the interval index and
  offset: x = log_r[j] - log_lam[i] is never formed.  Instead the host splits
  the scaled coordinates u_ij = (x_ij - knots[0]) / h = a_j + b_i (both >= 0,
  computed in float64) into integer and fractional parts, so the device only
  adds two fractions in [0, 1), which is accurate to ~1e-7 of a knot
  interval.  The sum over radii is an elementwise multiply + reduction, not a
  dot, so it can never be demoted to TF32.

* ``get_Qext`` replaces _mie_multi_x.get_Qext (Kitzmann & Heng 2017) on
  accelerators.  The sequential recursions there (continued fraction,
  downward A_n, upward C_n, and the running product B_n) take ~3000
  dependent numpy steps for large particles.  Each step is a Moebius map
  (or a multiplication), so they are evaluated as chunked scans: products
  of 2x2 matrices within short chunks, an associative scan across chunks,
  then the reference recurrence itself inside every chunk in parallel (see
  below).  It runs in float64/complex128 (under a local jax.enable_x64
  context) because the recursions are not accurate in float32, and it
  reproduces the reference implementation, including its continued-fraction
  stopping rule and per-x iteration counts (agreement ~1e-12).

Static shapes are bucketed (recursion length: 256, 1024, then powers of 2;
continued-fraction blocks: 64 or 1024 terms; number of x values: >= 512) so that
a whole retrieval needs only a handful of compiles (~2-5 s each).
"""
import functools
import math

import numpy as np
import scipy.interpolate
import jax
import jax.numpy as jnp
from jax import lax

from . import _mie_multi_x


def _pow2_at_least(n, minimum):
    return max(minimum, 1 << int(math.ceil(math.log2(max(n, 1)))))


def qext_on_device():
    """Whether get_Qext should run here rather than in numpy: only on
    accelerators.  (On a CPU backend the reference numpy recursion is
    faster than the float64 scans; both agree to ~1e-12.)"""
    return jax.default_backend() != "cpu"


def to_device(arrays):
    return tuple(jax.device_put(a) for a in arrays)


# ----------------------------------------------------------------------
# Spline evaluation + weighted sum over radii (float32)
# ----------------------------------------------------------------------
@jax.jit
def _spline_sum_kernel(ib, fb, ia, fa, coef, wts, n_int):
    s = fb[:, None] + fa[None, :]                  # in [0, 2)
    carry = jnp.floor(s)
    t = s - carry                                  # exact
    idx = ib[:, None] + ia[None, :] + carry.astype(jnp.int32)
    # Match the reference: indices past the last interval are clipped to it
    # and the cubic is extrapolated (only u == n_int exactly, in practice)
    over = jnp.maximum(idx - (n_int - 1), 0)
    idx = idx - over
    t = t + over.astype(t.dtype)
    c0, c1, c2, c3 = coef[0][idx], coef[1][idx], coef[2][idx], coef[3][idx]
    r = ((c0 * t + c1) * t + c2) * t + c3
    return jnp.sum(r * wts[None, :], axis=1)


def split_int_frac(u):
    """Split nonnegative float64 coordinates into int32 and float32 parts."""
    i = np.floor(u)
    return i.astype(np.int32), (u - i).astype(np.float32)


def spline_lognormal_integral(ib_fb, a, knot_values, weights):
    """out[i] = sum_j weights[j] * S(a[j] + b[i]), where S is the not-a-knot
    cubic spline through (k, knot_values[k]), k = 0..n_int, in units of the
    (uniform) knot spacing, and b[i] is given pre-split as ib_fb (device
    arrays from split_int_frac).  a: float64 (R,), >= 0.  Returns float64
    numpy (L,)."""
    n_int = len(knot_values) - 1
    # Unit-spacing coefficients (highest power first); tiny host solve
    coef = scipy.interpolate.CubicSpline(
        np.arange(n_int + 1, dtype=np.float64), knot_values).c
    n_pad = _pow2_at_least(n_int, 64)
    coef_p = np.zeros((4, n_pad), np.float32)
    coef_p[:, :n_int] = coef
    # Normalize weights to max 1 (results are rescaled in float64), keeping
    # the float32 kernel far from the subnormal range
    wscale = np.max(np.abs(weights))
    if wscale == 0 or not np.isfinite(wscale):
        wscale = 1.0
    ia, fa = split_int_frac(a)
    ib, fb = ib_fb
    out = _spline_sum_kernel(ib, fb, ia, fa, coef_p,
                             (weights / wscale).astype(np.float32),
                             np.int32(n_int))
    return np.asarray(out, dtype=np.float64) * wscale


# ----------------------------------------------------------------------
# Mie efficiencies (float64; chunked scans of Moebius maps)
# ----------------------------------------------------------------------
# Each recursion of the reference get_Qext is x_{i+1} = F_i(x_i) with F_i a
# Moebius map (or, for B_n, a multiplication).  Rows are split into c chunks
# of b = 8 rows.  Phase 1 composes each chunk's maps as a 2x2 matrix product
# (matrices generated on the fly; one fused elementwise kernel); phase 2
# carries the boundary values across chunks with an associative scan of the
# (c, n) chunk matrices (log depth, small arrays); phase 3 reruns the
# reference recurrence itself inside every chunk in parallel (one fused
# kernel).  Short unrolled chunks keep compile times and kernel counts low,
# and phase 3 does the same arithmetic as the reference.

_CHUNK = 8


def _chunking(N):
    b = min(_CHUNK, N)
    return b, N // b


def _combine_left(x, y):
    """normalize(y @ x) for associative_scan."""
    return _normalize(_matmul(y, x))


def _combine_right(x, y):
    """normalize(x @ y) for associative_scan."""
    return _normalize(_matmul(x, y))


def _shift_exclusive(S, first, reverse=False):
    """Exclusive scan from an inclusive one: rows shifted by one, with
    `first` (a tuple of (n,) arrays) at the start (end, if reverse)."""
    if reverse:
        return tuple(jnp.concatenate([s_[1:], f[None]], axis=0)
                     for s_, f in zip(S, first))
    return tuple(jnp.concatenate([f[None], s_[:-1]], axis=0)
                 for s_, f in zip(S, first))


def _matmul(x, y):
    x00, x01, x10, x11 = x
    y00, y01, y10, y11 = y
    return (x00 * y00 + x01 * y10, x00 * y01 + x01 * y11,
            x10 * y00 + x11 * y10, x10 * y01 + x11 * y11)


def _absmax(v):
    if jnp.iscomplexobj(v):
        return jnp.maximum(jnp.abs(v.real), jnp.abs(v.imag))
    return jnp.abs(v)


def _normalize(P):
    """Rescale a Moebius matrix by its largest component (projective)."""
    s = jnp.maximum(jnp.maximum(_absmax(P[0]), _absmax(P[1])),
                    jnp.maximum(_absmax(P[2]), _absmax(P[3])))
    r = 1.0 / jnp.where(s > 0, s, 1.0)
    return tuple(p * r for p in P)


def _mobius(P, v):
    return (P[0] * v + P[1]) / (P[2] * v + P[3])


def _eye_like(v):
    one = jnp.ones_like(v)
    zero = jnp.zeros_like(v)
    return (one, zero, zero, one)


def _rows(c, b, j):
    """Row indices c*b + j of chunk starts, as a (c, 1) column."""
    return (jnp.arange(c) * b + j)[:, None]


@functools.partial(jax.jit, static_argnums=(2,))
def _continued_fraction(z, nstart, K, k0, H0):
    """One block of K terms (k = k0+1 .. k0+K) of get_An's continued
    fraction for A_n(z) at order n = nstart (z: (n,)), with its stopping
    rule: the first k >= 2 at which the successive Lentz ratios
    h_k/h_{k-1} and q_k/q_{k-1} agree (np.allclose) for all z.  Convergents
    h_k/q_k come from H_k = T_1 ... T_k, T_k = [[a_k, 1], [1, 0]]; H0 is
    H_{k0} (normalized), so blocks chain; None means the identity (k0 = 0).
    Returns (A, converged, H_{k0+K})."""
    rdt = z.real.dtype
    if H0 is None:
        H0 = _eye_like(z)
    nu = nstart + 0.5
    b, c = _chunking(K)
    zc = jnp.broadcast_to(z[None, :], (c, z.shape[0]))

    def k_of(j):
        return _rows(c, b, j) + 1 + k0                # int (c, 1)

    def a_of(j):
        k = k_of(j)
        sign = jnp.where(k % 2 == 1, 1.0, -1.0).astype(rdt)
        return sign * 2 * (nu + k.astype(rdt) - 1) / zc

    # Phase 1: chunk products G_c = T_{k0+cb+1} ... T_{k0+cb+b}
    G = _eye_like(zc)
    for j in range(b):
        a = a_of(j)
        G = (G[0] * a + G[1], G[0], G[2] * a + G[3], G[2])  # G @ T
        if j % 4 == 3:
            G = _normalize(G)
    # Phase 2: H entering chunk c = H0 G_0 ... G_{c-1}
    S = lax.associative_scan(_combine_right, _normalize(G), axis=0)
    H0b = tuple(jnp.broadcast_to(h[None, :], (c, z.shape[0])) for h in H0)
    H = _normalize(_matmul(H0b, _shift_exclusive(S, _eye_like(z))))
    H_end = _normalize(_matmul(H0, tuple(s_[-1] for s_ in S)))
    # Phase 3: Wallis recurrences inside each chunk
    rat, close = [], []
    for j in range(b):
        a = a_of(j)
        H = _normalize((H[0] * a + H[1], H[0], H[2] * a + H[3], H[2]))
        num = H[0] / H[1]
        den = H[2] / H[3]
        ok = jnp.all(jnp.abs(num - den) <= 1e-8 + 1e-5 * jnp.abs(den), axis=1)
        close.append(ok & (k_of(j)[:, 0] >= 2))
        rat.append(H[0] / H[2])
    close = jnp.stack(close, axis=1).reshape(K)
    rat = jnp.stack(rat, axis=1).reshape(K, z.shape[0])
    kstar = jnp.argmax(close)
    return -nstart / z + rat[kstar], jnp.any(close), H_end


def _downward(z, s, max_iter, N):
    """A_i = f - 1/(f + A_{i+1}), f = (i+1)/z, for i = max_iter-2 .. 0, from
    A_{max_iter-1} = s; rows >= max_iter-1 hold s.  Returns (N, n)."""
    rdt = z.real.dtype
    b, c = _chunking(N)
    zc = jnp.broadcast_to(z[None, :], (c, z.shape[0]))

    def f_act(j):
        i = _rows(c, b, j)
        return (i.astype(rdt) + 1) / zc, i <= max_iter - 2

    # Phase 1: T_c = M_{cb} ... M_{cb+b-1}, M_i = [[f, f^2 - 1], [1, f]]
    T = _eye_like(zc)
    for j in range(b):
        f, act = f_act(j)
        M = (jnp.where(act, f, 1), jnp.where(act, f * f - 1, 0),
             jnp.where(act, 1, 0), jnp.where(act, f, 1))
        T = _matmul(T, tuple(m.astype(zc.dtype) for m in M))
        if j % 4 == 3:
            T = _normalize(T)
    # Phase 2: U_c = A_{(c+1)b} = (T_{c+1} ... T_{c-1 last})(s)
    v = jnp.broadcast_to(s, zc.shape).astype(zc.dtype)
    S = lax.associative_scan(_combine_left, _normalize(T), axis=0, reverse=True)
    S = _shift_exclusive(S, _eye_like(z), reverse=True)
    A = _mobius(S, v)
    # Phase 3: the reference recurrence inside each chunk
    out = [None] * b
    for j in reversed(range(b)):
        f, act = f_act(j)
        A = jnp.where(act, f - 1.0 / (f + A), A)
        out[j] = A
    return jnp.stack(out, axis=1).reshape(N, z.shape[0])


@functools.partial(jax.jit, static_argnums=(5,))
def _qext_main(xs, m, num_iters, max_iter, starts, N):
    s_mx, s_x = starts
    cdt = m.dtype
    rdt = xs.dtype
    n = xs.shape[0]
    xc = xs.astype(cdt)
    A_mx = _downward(m * xc, s_mx, max_iter, N)
    A_x = _downward(xs, s_x, max_iter, N)          # real for real x
    b, c = _chunking(N)
    A_mx = A_mx.reshape(c, b, n)
    A_x = A_x.reshape(c, b, n)
    xb = jnp.broadcast_to(xs[None, :], (c, n))

    def g_act(j):
        i = _rows(c, b, j)
        return i.astype(rdt) / xb, i >= 2, i

    sin, cos = jnp.sin(xs), jnp.cos(xs)
    B1 = 1.0 / (1 + 1j * (cos + xs * sin) / (sin - xs * cos))
    C1 = -1.0 / xc + 1.0 / (1.0 / xc + 1.0j)

    # Upward C_i = -g + 1/(g - C_{i-1}), g = i/x (i >= 2; C_0 = C_1 = C1).
    # Phase 1: chunk maps N_{cb+b-1} ... N_{cb}, N_i = [[g, 1-g^2], [-1, g]]
    T = _eye_like(xb.astype(cdt))
    for j in range(b):
        g, act, _ = g_act(j)
        Nm = (jnp.where(act, g, 1), jnp.where(act, 1 - g * g, 0),
              jnp.where(act, -1, 0), jnp.where(act, g, 1))
        T = _matmul(tuple(q.astype(cdt) for q in Nm), T)
        if j % 4 == 3:
            T = _normalize(T)
    # Phase 2: L_c = C entering chunk c = (U_{c-1} ... U_0)(C1)
    S = lax.associative_scan(_combine_left, _normalize(T), axis=0)
    S = _shift_exclusive(S, _eye_like(C1))
    L = _mobius(S, jnp.broadcast_to(C1[None, :], (c, n)))

    def c_and_r(j, C):
        g, act, i = g_act(j)
        C = jnp.where(act, -g + 1.0 / (g - C), C)
        r = jnp.where(act, (C + g) / (A_x[:, j] + g), 1.0)
        return C, r

    # B_i = B_{i-1} (C_i + g)/(A_x[i] + g): chunk products, then prefixes
    C = L
    R = jnp.ones_like(L)
    for j in range(b):
        C, r = c_and_r(j, C)
        R = R * r
    LB = B1.astype(cdt)[None, :] * jnp.concatenate(
        [jnp.ones((1, n), cdt), jnp.cumprod(R, axis=0)[:-1]], axis=0)

    # Final pass: a_n, b_n and the masked sum, chunk by chunk
    C, B = L, LB
    acc = jnp.zeros((c, n), rdt)
    for j in range(b):
        C, r = c_and_r(j, C)
        B = B * r
        _, _, i = g_act(j)
        Amx = A_mx[:, j]
        Ax = A_x[:, j]
        an = B * (Amx / m - Ax) / (Amx / m - C)
        bn = B * (Amx * m - Ax) / (Amx * m - C)
        use = (i >= 1) & (i < num_iters[None, :])
        acc = acc + jnp.where(use, (2 * i.astype(rdt) + 1) * (an + bn).real, 0.0)
    return jnp.sum(acc, axis=0) * 2 / xs**2


def _num_rows(max_iter):
    """Static recursion length.  Few buckets (each is a compile, ~2-5 s):
    on a GPU, padding rows is nearly free at these sizes."""
    if max_iter <= 256:
        return 256
    if max_iter <= 1024:
        return 1024
    return _pow2_at_least(max_iter, 2048)


def _cf_block(zmax, max_iter):
    """Continued-fraction block length: it converges within tens of terms
    unless |z| exceeds the order (weakly absorbing large particles), when
    it can take ~|z| - n terms.  Two fixed sizes, chained as needed."""
    return 64 if 1.25 * max(zmax - max_iter, 0) + 32 <= 64 else 1024


def _cf_start(z_d, ns_d, K, k0_d):
    """Launch the first continued-fraction block (no host sync)."""
    return _continued_fraction(z_d, ns_d, K, k0_d, None)


def get_Qext(m, xs):
    """Drop-in replacement for _mie_multi_x.get_Qext (float64 numpy out)."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return np.zeros(0)
    num_iters = _mie_multi_x.get_iterations_required(xs)
    max_iter = max(int(np.max(num_iters)), 1)
    # A fixed minimum width, so that partial cache hits (any number of
    # misses) don't each compile a new variant
    n_pad = _pow2_at_least(n, 512)
    # Pad with the smallest x (cheap; results discarded)
    xs_p = np.full(n_pad, np.min(xs))
    xs_p[:n] = xs
    it_p = np.full(n_pad, num_iters.min(), dtype=np.int32)
    it_p[:n] = num_iters
    N = _num_rows(max_iter)
    K_mx = _cf_block(abs(m) * np.max(xs), max_iter)
    K_x = _cf_block(np.max(xs), max_iter)
    m = complex(m)
    with jax.enable_x64(True):
        x_d, mx_d, m_d, it_d, mi_d, ns_d, k0_d = jax.device_put((
            xs_p, m * xs_p, np.complex128(m), it_p, np.int32(max_iter),
            np.float64(max_iter - 1), np.int32(0)))
        cf = {"mx": [mx_d, K_mx, 0, _cf_start(mx_d, ns_d, K_mx, k0_d)],
              "x": [x_d, K_x, 0, _cf_start(x_d, ns_d, K_x, k0_d)]}
        while True:
            Q = _qext_main(x_d, m_d, it_d, mi_d,
                           (cf["mx"][3][0], cf["x"][3][0]), N)
            Q, ok_mx, ok_x = jax.device_get((Q, cf["mx"][3][1], cf["x"][3][1]))
            if ok_mx and ok_x:
                break
            # Not converged within the terms so far: continue the
            # continued fraction from where it stopped (rare)
            for name, ok in (("mx", ok_mx), ("x", ok_x)):
                if ok:
                    continue
                z_d, K, k0, (_, _, H_end) = cf[name]
                k0 += K
                if k0 >= 1 << 17:
                    raise RuntimeError("Mie continued fraction did not converge")
                K = 1024
                cf[name] = [z_d, K, k0, _continued_fraction(
                    z_d, ns_d, K, jnp.int32(k0), H_end)]
        return np.asarray(Q, dtype=np.float64)[:n]
