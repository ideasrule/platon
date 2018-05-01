import unittest
import tau_los
import numpy as np
import scipy.interpolate

class TestTauLOS(unittest.TestCase):

    def test_realistic(self):
        absorption_coeffs = np.loadtxt("testing_data/exotransmit_kappa")
        heights = np.loadtxt("testing_data/exotransmit_heights")
        expected_tau = np.loadtxt("testing_data/exotransmit_tau")

        tau = tau_los.get_line_of_sight_tau(absorption_coeffs, heights)
        
        self.assertTrue(np.allclose(tau, expected_tau))

    def test_dl(self):
        radii = np.array([124,100,44,33,10,3.1,2.45])
        dl = tau_los.get_dl(radii)

        for i in range(dl.shape[0]):
            for j in range(dl.shape[1]):
                r_prime = radii[i+1]
                r_prime_higher = radii[i]
                                
                r = radii[j+1]
                dist_higher = 2*np.sqrt(r_prime_higher**2 - r**2)
                dist_lower = 2*np.sqrt(r_prime**2 - r**2)
                if r > r_prime: self.assertEquals(dl[i][j], 0)
                else: self.assertEquals(dl[i][j], dist_higher - dist_lower)


    def test_analytic_simple(self):
        absorption_coeff = np.ones((12,100))
        Rp = 1000
        radii = Rp + np.linspace(0, 100, 101)
        radii = np.flip(radii, 0)
        tau = tau_los.get_line_of_sight_tau(absorption_coeff, radii)
        expected_tau = 2*np.sqrt(radii[0]**2 - radii[1:]**2)
        for t in tau:
            np.allclose(t, expected_tau)

    def test_analytic_exponential(self):
        absorption_coeff = np.ones((1, 1000))
        Rp = 2000
        scale_height = 100
        radii = Rp + np.linspace(0, 1000, 1001)
        radii = np.flip(radii, 0)
        analytic_tau = []
        for i, r in enumerate(radii[1:]):
            absorption_coeff[:,i] *= np.exp(-(r-Rp)/scale_height)
            analytic_tau.append(2*scipy.integrate.quad(lambda x: np.exp(-(x-Rp)/scale_height)*x/np.sqrt(x**2 - r**2), r, radii[0])[0])
        tau = tau_los.get_line_of_sight_tau(absorption_coeff, radii)
        analytic_tau = np.array(analytic_tau)
        rel_diff = (tau - analytic_tau)/analytic_tau
        self.assertTrue(np.all(rel_diff < 0.01))
        
        #print tau[0,0:100], analytic_tau[0:100]
            
        
    
        
if __name__ == '__main__':
    unittest.main()
