FORCE_CPU = True  # force to use CPU if this is True

try:
    if not FORCE_CPU:
        from cupy import *
        from cupyx import scipy
        print("GPU acceleration enabled")
    else:
        raise ImportError
except ImportError:
    if not FORCE_CPU:
        print("cupy not found. Disabling GPU acceleration")
    from numpy import *
    import scipy
    import scipy.special

def cpu(arr):
    try:
        return arr.get()
    except:
        return arr
