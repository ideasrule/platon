from __future__ import print_function

import numpy as np
import scipy.interpolate

class MieCache:
    def __init__(self):
        self.interpolator = None
        self.all_xs = np.array([])
        self.all_Qexts = np.array([])
        self.all_ms = np.array([], dtype=complex)
        self.total_gets = 1
        self.total_hits = 1
        
    def get(self, m, xs, max_frac_error=0.01):
        self.total_gets += 1
        #if self.total_gets % 10 == 0:
        #    print float(self.total_hits)/self.total_gets, len(self.all_xs)
        if len(self.all_xs) == 0:
            return None
        
        closest_matches = np.searchsorted(self.all_xs, xs)
        if np.min(closest_matches) == 0 or np.max(closest_matches) == len(self.all_xs):
            return None

        if np.any(self.all_ms[closest_matches] != m):
            return None

        frac_errors = np.abs(self.all_xs[closest_matches] - xs)/xs
        if np.max(frac_errors) > max_frac_error:
            return None

        Qexts = self.interpolator(xs)
        self.total_hits += 1
        return Qexts


    def add(self, m, xs, Qexts, size_limit=1000000):
        print("Adding")
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

'''import numpy as np
import time
import scipy.interpolate

cache = MieCache()
while True:
    xs = np.random.uniform(0, 1000, 1000)
    y_vals = np.random.uniform(0, 2, 1000)
    cache.add(xs, y_vals)
    print cache.get(np.array([1, 3, 951]))

#cache = MieCache()
#cache.add(xs, y_vals)

exit(0)


start = time.time()
interpolator = scipy.interpolate.interp1d(arr, y_vals)
interpolator(152)
#np.sort(arr)
print time.time() - start
print len(arr)
'''
