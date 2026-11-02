from collections import OrderedDict

import numpy as np
from . import _mie_multi_x


class MieCache:
    """Cache of Qext(x) per refractive index m.

    Entries are stored per m (sorted by x), so lookups and insertions cost
    O(entries for that m) rather than O(all entries): in a retrieval that
    fits the refractive index, every call brings a new m, and a single flat
    array of all entries (re-sorted on every insertion) grew to cost ~60 ms
    per call once it held 1e6 entries.  When the total exceeds size_limit,
    the least recently used refractive indices are dropped first."""

    def __init__(self):
        self._store = OrderedDict()   # m -> (sorted xs, Qexts)
        self._size = 0

    def __len__(self):
        return self._size

    def get_from_cache(self, m, xs, max_frac_error=0.05):
        xs = np.asarray(xs, dtype=float)
        result = np.ones(len(xs)) * np.nan
        entry = self._store.get(m)
        if entry is None:
            return result
        self._store.move_to_end(m)
        cached_xs, cached_Qexts = entry

        in_cache = np.ones(len(xs), dtype=bool)
        closest_matches = np.searchsorted(cached_xs, xs)
        in_cache[np.logical_or(closest_matches == 0,
                               closest_matches == len(cached_xs))] = False
        closest_matches[closest_matches == len(cached_xs)] -= 1
        frac_errors = np.abs(cached_xs[closest_matches] - xs) / xs
        in_cache[frac_errors > max_frac_error] = False

        result[in_cache] = np.interp(xs[in_cache], cached_xs, cached_Qexts)
        return result

    def get_and_update(self, m, xs, qext_func=None):
        # Get from cache if available, from Mie calculations if not.
        # Put results of Mie calculations into cache.  qext_func(m, xs)
        # computes the misses (default: _mie_multi_x.get_Qext).
        if qext_func is None:
            qext_func = _mie_multi_x.get_Qext
        Qexts = self.get_from_cache(m, xs)
        cache_misses = np.isnan(Qexts)
        if np.sum(cache_misses) > 0:
            Qexts[cache_misses] = qext_func(m, xs[cache_misses])
            self.add(m, xs[cache_misses], Qexts[cache_misses])
        return Qexts

    def add(self, m, xs, Qexts, size_limit=1000000):
        if len(xs) == 0:
            return
        xs = np.asarray(xs, dtype=float)
        Qexts = np.asarray(Qexts, dtype=float)
        old = self._store.pop(m, None)
        if old is not None:
            self._size -= len(old[0])
            xs = np.append(old[0], xs)
            Qexts = np.append(old[1], Qexts)
        if len(xs) > size_limit:
            keep = np.sort(np.random.choice(len(xs), size_limit - 1,
                                            replace=False))
            xs, Qexts = xs[keep], Qexts[keep]
        # Stable sort: O(n) on the (sorted old + new) concatenation
        p = np.argsort(xs, kind="stable")
        self._store[m] = (xs[p], Qexts[p])
        self._size += len(xs)
        while self._size > size_limit and len(self._store) > 1:
            _, (old_xs, _) = self._store.popitem(last=False)
            self._size -= len(old_xs)
