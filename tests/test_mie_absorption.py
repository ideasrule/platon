import unittest
import numpy as np
import scipy.integrate
from nose.tools import nottest

from platon import mie_multi_x
from platon.transit_depth_calculator import TransitDepthCalculator

class TestMieAbsorption(unittest.TestCase):
    def exact_cross_section(self, r_mean, wavelength, ri, sigma=0.5):        
        def integrand(y):
            r = r_mean * np.exp(np.sqrt(2) * sigma * y)
            x = 2*np.pi*r/wavelength
            if x < 1e-6: Qext = 0
            elif x > 1e4:
                # Optimization needed, or else code takes forever
                Qext = 2
            else:
                Qext = mie_multi_x.get_Qext(ri, [x])[0]                    
            return np.exp(-y**2) * np.sqrt(np.pi) * r_mean**2 * np.exp(2*np.sqrt(2)*sigma*y) * Qext

        result, error = scipy.integrate.quad(integrand, -10, 10)
        return result

    @nottest
    def run_test(self, m, r_mean, sigma, frac_scale_height=2):
        n_0 = 2.3
        calc = TransitDepthCalculator()
        # Calculating the exact solution for all wavelengths would take
        # forever, so we only do a few
        to_include = np.array([1, 10, 100, 1000, 3000, 4600])
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
        self.assertTrue(np.allclose(rough_cross_sections[to_include], exact_cross_sections))

        for i in range(absorption.shape[1]):
            P = calc.P_grid[P_cond][i]
            ref_P = np.max(calc.P_grid[P_cond])
            ratio = (P/ref_P)**(1.0/frac_scale_height)
            self.assertTrue(np.allclose(rough_cross_sections * n_0 * ratio, absorption[:, i, :].flatten()))
            
            
    def test_complex(self):
        self.run_test(1.33-0.1j, 1e-6, 0.5)
        self.run_test(2-2j, 1e-5, 0.25)

        
    def test_real(self):
        self.run_test(1.7, 1e-6, 0.5)
        self.run_test(1.3, 2e-5, 0.65)
        
if __name__ == '__main__':
    unittest.main()
