import scipy.special
import numpy as np
import sys
import matplotlib.pyplot as plt

def get_k_coeffs(absorb_coeffs, binsize=200, n_gauss=10):
    points, weights = scipy.special.roots_legendre(n_gauss)
    percentiles = 100 * (points + 1) / 2
    k_coeffs = []
    for i in range(absorb_coeffs.shape[2] // binsize):
        vals = absorb_coeffs[:,:,i*binsize : (i+1)*binsize]
        k_coeffs.append(np.percentile(vals, percentiles, axis=2))
    k_coeffs = np.array(k_coeffs)
    k_coeffs = k_coeffs.reshape((k_coeffs.shape[0] * k_coeffs.shape[1], k_coeffs.shape[2], k_coeffs.shape[3]))
    k_coeffs = k_coeffs.transpose((1,2,0))
    return k_coeffs

    print(k_coeffs.shape)
    plt.loglog(k_coeffs[20,7])
    plt.show()


wavelengths = 1e-6 * np.exp(np.arange(np.log(0.2), np.log(30), 1./20000))
k_wavelengths = np.repeat(wavelengths[::200][:-1], 10)
for filename in sys.argv[1:]:
    output_filename = filename.replace("absorb_coeffs_", "k_coeffs_")
    absorb_coeffs = np.load(filename)
    k_coeffs = get_k_coeffs(absorb_coeffs)
    np.save("k_wavelengths.npy", k_wavelengths)
    np.save(output_filename, k_coeffs)
