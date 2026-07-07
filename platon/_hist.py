import numpy as np


def _hist_bin_fd(x):
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    return 2 * iqr * x.size**(-1. / 3)


def _hist_bin_sturges(x, lo, hi):
    return (hi - lo) / (np.log2(x.size) + 1.)


def _hist_bin_auto(x, lo, hi):
    fd_bw = _hist_bin_fd(x)
    sturges_bw = _hist_bin_sturges(x, lo, hi)
    if fd_bw > 0:
        return min(fd_bw, sturges_bw)
    else:
        return sturges_bw


def get_num_bins(x, lo, hi):
    """Number of histogram bins (numpy's 'auto' rule).  `lo`/`hi` are the
    extremes of x, passed in because the caller already knows them (skipping
    two full passes over x)."""
    width = _hist_bin_auto(x, lo, hi)
    n = int(np.ceil((hi - lo) / width))
    return n
