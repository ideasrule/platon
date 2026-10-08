FORCE_CPU = False  # force to use CPU if this is True

if FORCE_CPU:
    print("forcing CPU")
    from numpy import *
    import scipy
    import scipy.special
    from scipy import interpolate, ndimage
else:
    try:
        import cupy as _cupy_test
        _cupy_test.zeros(1)  # trigger CUDA init to catch runtime failures early
        del _cupy_test
        from cupy import *
        from cupyx import scipy
        from cupyx.scipy import interpolate, ndimage
    except Exception:
        print("cupy not available or no GPU found. Disabling GPU acceleration")
        from numpy import *
        import scipy
        import scipy.special
        from scipy import interpolate, ndimage

def cpu(arr):
    try:
        return arr.get()
    except:
        return arr
