from __future__ import print_function

import numpy as np
import scipy.interpolate

from . import mie_multi_x

class MieCache:
    def __init__(self):
        self.interpolator = None
        self.all_xs = np.array([])
        self.all_Qexts = np.array([])
        self.all_ms = np.array([], dtype=complex)

        
    def get_from_cache(self, m, xs, max_frac_error=0.05):
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
        
    def get_and_update(self, m, xs):
        # Get from cache if available, from Mie calculations if not. 
        # Put results of Mie calculations into cache.
        Qexts = self.get_from_cache(m, xs)
        cache_misses = np.isnan(Qexts)
        if np.sum(cache_misses) > 0:
            Qexts[cache_misses] = mie_multi_x.get_Qext(m, xs[cache_misses])
            self.add(m, xs[cache_misses], Qexts[cache_misses])
        return Qexts

    
    def add(self, m, xs, Qexts, size_limit=1000000):
        if len(xs) == 0:
            return
        
        self.all_xs = np.append(self.all_xs, xs)
        self.all_Qexts = np.append(self.all_Qexts, Qexts)
        self.all_ms = np.append(self.all_ms, np.array([m] * len(xs)))
        if len(self.all_xs) > size_limit:
            to_remove = np.random.choice(
                range(len(self.all_xs)), len(self.all_xs) - size_limit + 1,
                replace=False)
            
            self.all_xs = np.delete(self.all_xs, to_remove)
            self.all_Qexts = np.delete(self.all_Qexts, to_remove)
            self.all_ms = np.delete(self.all_ms, to_remove)

        p = np.argsort(self.all_xs)
        self.all_xs = self.all_xs[p]
        self.all_Qexts = self.all_Qexts[p]
        self.all_ms = self.all_ms[p]
        
        self.interpolator = scipy.interpolate.interp1d(
            self.all_xs[self.all_ms == m],
            self.all_Qexts[self.all_ms == m],
            assume_sorted=True)

