from interpolator_3D import fast_interpolate
import unittest
import numpy as np

class TestInterpolation(unittest.TestCase):
    def test_one_x(self):
        ny = 2
        nx = 1
        data = np.array([4, 6]).reshape((1, ny, nx))
        grid_x = np.array([100])
        grid_y = np.array([1, 2])
        target_x = np.array([100, 100, 100])
        target_y = np.array([1.1, 1.2, 1.3])
        
        result = fast_interpolate(data, grid_x, grid_y, target_x, target_y)
        expected_result = np.array([(target_y-1) * (6-4) + 4])
        self.assertTrue(np.allclose(result, expected_result))


    def test_one_y(self):
        ny = 1
        nx = 2
        data = np.array([4, 6]).reshape((1, ny, nx))
        grid_x = np.array([100, 200])
        grid_y = np.array([1])
        target_x = np.array([100, 150, 190, 200])
        target_y = np.array([1, 1, 1, 1])
        
        result = fast_interpolate(data, grid_x, grid_y, target_x, target_y)
        print result
        expected_result = np.array([[4, 5, 5.8, 6]])
        self.assertTrue(np.allclose(result, expected_result))
        

if __name__ == '__main__':
    unittest.main()
