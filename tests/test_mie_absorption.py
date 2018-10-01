import unittest
import numpy as np
import scipy.integrate
from nose.tools import nottest
import matplotlib.pyplot as plt

from platon import _mie_multi_x
from platon.transit_depth_calculator import TransitDepthCalculator

class TestMieAbsorption(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMieAbsorption, self).__init__(*args, **kwargs)

        # We're storing this object to take advantage of its cache, not
        # for speed, but to test the cache
        self.calc = TransitDepthCalculator()
        
    def exact_cross_section(self, r_mean, wavelength, ri, sigma=0.5):        
        def integrand(y):
            r = r_mean * np.exp(np.sqrt(2) * sigma * y)
            x = 2*np.pi*r/wavelength
            if x < 1e-6: Qext = 0
            elif x > 1e4:
                # Optimization needed, or else code takes forever
                Qext = 2
            else:
                Qext = _mie_multi_x.get_Qext(ri, [x])[0]                    
            return np.exp(-y**2) * np.sqrt(np.pi) * r_mean**2 * np.exp(2*np.sqrt(2)*sigma*y) * Qext

        result, error = scipy.integrate.quad(integrand, -5, 5, epsrel=1e-3, epsabs=0, limit=100)
        return result

    @nottest
    def run_test(self, m, r_mean, sigma, frac_scale_height=2, rtol=1e-5, atol=1e-5):
        n_0 = 2.3e9
        calc = self.calc #TransitDepthCalculator()
        # Calculating the exact solution for all wavelengths would take
        # forever, so we only do a few
        to_include = np.array([1, 200, 1000, 3000, 4600])
        #to_include = np.array([4600])
        exact_cross_sections = [self.exact_cross_section(
            r_mean, w, m, sigma) for w in calc.lambda_grid[to_include]]

        # This technically gets us absorption cross sections, but for n_0=1 and
        # a single pressure in the list, this should equal cross section
        P_cond = calc.P_grid <= 1e5
        absorption = calc._get_mie_scattering_absorption(
            P_cond, calc.T_grid == 800, m, r_mean,
            frac_scale_height, n_0, sigma=sigma)

        # absorption at max pressure for n_0=1 should equal cross section
        rough_cross_sections = absorption[:, -1, :].flatten()/n_0
        frac_dev = np.abs(rough_cross_sections[to_include] - exact_cross_sections)/exact_cross_sections
        #print(frac_dev)
        self.assertTrue(np.max(frac_dev) < 0.01)

        for i in range(absorption.shape[1]):
            P = calc.P_grid[P_cond][i]
            ref_P = np.max(calc.P_grid[P_cond])
            ratio = (P/ref_P)**(1.0/frac_scale_height)
            self.assertTrue(np.allclose(rough_cross_sections * n_0 * ratio, absorption[:, i, :].flatten(), atol=0, rtol=1e-3))
            
            
    def test_complex(self):
        self.run_test(1.33-0.1j, 1e-6, 0.5)
        self.run_test(1.33-0.1j, 2e-6, 0.3)
        self.run_test(1.33-0.1j, 1e-5, 0.4)
        self.run_test(2-2j, 1e-5, 0.25)

    @unittest.skip("For some reason integration stalls on OS X VMs; not sure why")        
    def test_real(self):
        self.run_test(1.7, 1e-6, 0.5)

        # Too slow in Travis CI, so commenting out
        self.run_test(1.3, 2e-5, 0.65)
        
if __name__ == '__main__':
    unittest.main()
