import unittest
import tau_los
import numpy as np

class TestTauLOS(unittest.TestCase):

    def test_realistic(self):
        absorption_coeffs = np.loadtxt("testing_data/exotransmit_kappa")
        heights = np.loadtxt("testing_data/exotransmit_heights")
        expected_tau = np.loadtxt("testing_data/exotransmit_tau")

        tau = tau_los.get_line_of_sight_tau(absorption_coeffs, heights)
        
        self.assertTrue(np.allclose(tau, expected_tau))

    def test_dl(self):
        heights = np.array([12, 11, 10])
        dl = tau_los.get_dl(heights)
        print dl
        for h_index in range(0, len(heights)):
            for h_prime_index in range(h_index):
                dist_higher = 2*np.sqrt(heights[h_prime_index]**2 - heights[h_index]**2)
                dist_lower = 2*np.sqrt(heights[h_prime_index+1]**2 - heights[h_index]**2)
                print h_index, h_prime_index
                
                print dist_higher - dist_lower, dl[h_prime_index, h_index-1]
                #print dist_lower - dist_higher, dl[h_index, h_prime_index]

if __name__ == '__main__':
    unittest.main()
