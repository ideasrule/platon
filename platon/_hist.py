import numpy as np


def _hist_bin_fd(x):
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return 2 * iqr * x.size**(-1. / 3)


def _hist_bin_sturges(x):
    return (x.max() - x.min()) / (np.log2(x.size) + 1.)


def _hist_bin_auto(x):
    fd_bw = _hist_bin_fd(x)
    sturges_bw = _hist_bin_sturges(x)
    if fd_bw > 0:
        return min(fd_bw, sturges_bw)
    else:
        return sturges_bw


def get_num_bins(x):
    width = _hist_bin_auto(x)
    n = int(np.ceil((x.max() - x.min()) / width))
    return n
