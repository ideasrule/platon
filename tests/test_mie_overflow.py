import unittest
import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile
from platon.constants import R_sun, R_jup, M_jup


class TestMieOverflow(unittest.TestCase):
    def test_thin_dense_aerosol_layer_is_finite(self):
        # A small aerosol scale height makes num_den * (P / P_ref)**(1 / fsh)
        # exceed the float32 maximum deep in the atmosphere.  The resulting
        # inf, multiplied by zero path lengths, used to turn every transit
        # depth into NaN.  Parameters are from a prior draw of a HATS-6b Mie
        # retrieval that crashed nautilus (a puffy, low-gravity planet).
        calc = TransitDepthCalculator()
        _, depths, _ = calc.compute_depths(
            Profile.isothermal(953.0), 0.6 * R_sun, 4.669e26, 9.217e7,
            logZ=1.8126, CO_ratio=0.5412, cloudtop_pressure=10**2.6349,
            ri=1.6 - 10**-4.0257 * 1j, part_size=10**-7.9811,
            number_density=10**12.259, frac_scale_height=0.168)
        self.assertTrue(np.all(np.isfinite(depths)))

    def test_cap_does_not_change_moderate_aerosols(self):
        # Away from overflow, the capped log-space evaluation must reproduce
        # the plain power law to float32 precision
        calc = TransitDepthCalculator()
        args = (Profile.isothermal(900), 0.6 * R_sun, 0.32 * M_jup, 1.05 * R_jup)
        kwargs = dict(logZ=0, CO_ratio=0.53, cloudtop_pressure=1e4,
                      ri=1.6 - 0.01j, part_size=1e-7)
        _, d_low, _ = calc.compute_depths(*args, number_density=1e8,
                                       frac_scale_height=1, **kwargs)
        _, d_none, _ = calc.compute_depths(*args, number_density=1e-30,
                                        frac_scale_height=1, **kwargs)
        self.assertTrue(np.all(np.isfinite(d_low)))
        # Aerosols must still add opacity
        self.assertGreater(np.mean(d_low), np.mean(d_none))


if __name__ == '__main__':
    unittest.main()
