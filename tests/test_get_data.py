import unittest
import os
import tempfile

from platon._get_data import get_data

class TestGetData(unittest.TestCase):
    def test_get_data(self):
        target_dir = tempfile.mkdtemp()
        get_data(target_dir)

        self.assertFalse(os.path.isfile(os.path.join(target_dir, "data.zip")))
        self.assertTrue(os.path.isdir(os.path.join(target_dir, "data")))

        expectedFiles = ["collisional_absorption.pkl", "pressures.npy", "species_info", "temperatures.npy", "wavelengths.npy", "stellar_spectra.pkl"]
        expectedDirs = ["Absorption", "abundances"]

        for f in expectedFiles:
            filename = os.path.join(target_dir, "data", f)
            self.assertTrue(os.path.isfile(filename))

        for d in expectedDirs:
            dirname = os.path.join(target_dir, "data", d)
            self.assertTrue(os.path.isdir(dirname))

if __name__ == '__main__':
    unittest.main()        
