import unittest

import jax
import jax.numpy as jnp
import numpy as np

from platon._forward_model import _hydrostatic
from platon._forward_prep import _pack_scalars
from platon.constants import AMU, G, M_jup, M_sun, R_jup, R_sun, k_B


class TestHydrostatic(unittest.TestCase):
    def test_jit_bound_check_matches_float64_reference(self):
        cases = [
            (M_jup, R_jup, 1000., 5700., False),
            (5.97e20, 6.378e6, 300., 6100., True),
            (M_jup, R_jup, 1000., 1000., True),
        ]
        compiled = jax.jit(_hydrostatic)
        for n in [20, 100, 1000]:
            pressures = np.logspace(-4, 8, n)
            for mass, radius, temperature, star_temp, expected in cases:
                with self.subTest(n=n, mass=mass, star_temp=star_temp):
                    scalars = jnp.asarray(_pack_scalars(
                        rs=R_sun, mp=mass, rp=radius,
                        t_star_hydro=star_temp, ref_pressure=1e5),
                        dtype=jnp.float32)
                    inputs = (scalars, jnp.asarray(pressures),
                              jnp.full(n, temperature), jnp.full(n, 2.3))
                    radius_estimate = 1 / (
                        1 / radius + k_B * temperature * np.log(1e-4 / 1e5) /
                        (G * mass * 2.3 * AMU))
                    hill = R_sun * (star_temp / temperature)**2 * \
                        (mass / (3 * M_sun))**(1 / 3)
                    self.assertEqual(radius_estimate < 0 or radius_estimate > hill,
                                     expected)
                    eager = _hydrostatic(*inputs)
                    actual = compiled(*inputs)
                    self.assertEqual(bool(eager[2]), expected)
                    self.assertEqual(bool(actual[2]), expected)
                    np.testing.assert_allclose(actual[0], eager[0], rtol=2e-4)
