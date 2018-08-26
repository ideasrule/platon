from __future__ import print_function

import numpy as np
import scipy.interpolate

class MieCache:
    def __init__(self):
        self.interpolator = None
        self.all_xs = np.array([])
        self.all_Qexts = np.array([])
        self.all_ms = np.array([], dtype=complex)

        
    def get(self, m, xs, max_frac_error=0.01):
        result = np.ones(len(xs)) * np.nan

        if len(self.all_xs) == 0:
            return result
        
        in_cache = np.ones(len(xs), dtype=bool)
        closest_matches = np.searchsorted(self.all_xs, xs)        
        in_cache[np.logical_or(closest_matches == 0, closest_matches == len(self.all_xs))] = False
        
        closest_matches[closest_matches == len(self.all_xs)] -= 1
        in_cache[self.all_ms[closest_matches] != m] = False
        frac_errors = np.abs(self.all_xs[closest_matches] - xs)/xs
        in_cache[frac_errors > max_frac_error] = False

        result[in_cache] = self.interpolator(xs[in_cache])
        
        return result
        

    def add(self, m, xs, Qexts, size_limit=1000000):
        if len(xs) == 0:
            return
        
        self.all_xs = np.append(self.all_xs, xs)
        self.all_Qexts = np.append(self.all_Qexts, Qexts)
        self.all_ms = np.append(self.all_ms, np.array([m] * len(xs)))
        if len(self.all_xs) > size_limit:
            to_remove = np.random.choice(
                range(len(self.all_xs)), len(self.all_xs) - size_limit,
                replace=True)
            
            self.all_xs = np.delete(self.all_xs, to_remove)
            self.all_Qexts = np.delete(self.all_Qexts, to_remove)
            self.all_ms = np.delete(self.all_ms, to_remove)

        p = np.argsort(self.all_xs)
        self.all_xs = self.all_xs[p]
        self.all_Qexts = self.all_Qexts[p]
        self.all_ms = self.all_ms[p]
        
        self.interpolator = scipy.interpolate.interp1d(
            self.all_xs, self.all_Qexts, assume_sorted=True)

