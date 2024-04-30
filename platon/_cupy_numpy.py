FORCE_CPU = False  # force to use CPU if this is True

if FORCE_CPU:
    print("forcing CPU")
    from numpy import *
    import scipy
    import scipy.special
    from scipy import interpolate
else:
    try:
        from cupy import *
        from cupyx import scipy
        from cupyx.scipy import interpolate
    except:
        print("cupy not found. Disabling GPU acceleration")
        from numpy import *
        import scipy
        import scipy.special
        from scipy import interpolate

def cpu(arr):
    try:
        return arr.get()
    except:
        return arr
