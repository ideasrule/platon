import matplotlib.pyplot as plt
import numpy as np

from platon.constants import h, c, k_B, R_jup, M_jup, R_sun, pc
from platon.flux_calculator import FluxCalculator
from platon.TP_profile import Profile
from scipy.ndimage import gaussian_filter

#Evaluated params: ln_prob=8.82e+03	Rp=1.65 R_jup	log_C3=-3.91 	log_C2=-7.08 	log_CO=-6.20 	T_irr=1419.64 	log_k_th=-0.92 	log_gamma=-0.16 	log_gamma2=-2.15 	alpha=0.06 	beta=1.56 	T_int=102.53 	log_cloudtop_P=3.73

p = Profile()
p.set_from_radiative_solution(1419.64, 1.5*M_jup, 1.65*R_jup, 1.56, -0.92, -0.16, log_gamma2=-2.15, alpha=0.06, T_int=102.53)
calc = FluxCalculator(method="xsec") #"ktables" for correlated k

#Uncomment below to get binned eclipse depths
#edges = np.linspace(1.1e-6, 1.7e-6, 30)
#bins = np.array([edges[0:-1], edges[1:]]).T
#calc.change_wavelength_bins(bins)

calc.change_wavelength_bins([[1.6e-6, 1.7e-6], [1.8e-6,1.9e-6]])

wavelengths, fluxes, info_dict = calc.compute_fluxes(
    1.65*R_jup / (720 * pc), p, 1.5*M_jup, 1.65*R_jup, full_output=True,
    cloudtop_pressure=10.**3.73,
    logZ=None, CO_ratio=None, gases=["C3", "C2", "CO", "He"], vmrs=[10**-3.91, 10**-7.08, 10**-6.20, 0.999])
plt.semilogy(p.get_temperatures(), p.get_pressures())
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (Pa)")
plt.gca().invert_yaxis()
plt.figure()


obs_waves, obs_fluxes, obs_errors = np.loadtxt("prism_pulsar_planet.txt", unpack=True)
plt.plot(obs_waves, obs_fluxes)


plt.plot(1e6*wavelengths, gaussian_filter(fluxes, 150))
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("Flux")
plt.show()
