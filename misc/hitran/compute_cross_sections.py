import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Voigt1D
import scipy.integrate
import pickle
from astropy.constants import h, c, k_B, u

nu, sw, A, trans_id, gamma_air, delta_air, n_air, nu_lower = np.loadtxt("h2o.out", skiprows=1, unpack=True)

all_wavenums = np.logspace(-np.log10(30e-4), -np.log10(0.3e-4), 1000000)
cross_sections = np.zeros(len(all_wavenums))


def get_cross_sections(all_wavenums, wavenum_ref, integral, P, T, gamma_ref, n_air, delta_ref, T_ref=296.0):
    P_atm = P/1e5 #Pascals to atm
    wavenum = wavenum_ref + delta_ref * P_atm
    gamma = (T_ref/T)**n_air * gamma_ref * P_atm

    #print wavenum_ref, delta_ref * P_atm, (T_ref/T)**n_air * P_atm, n_air
    #print gamma_ref, gamma
    
    fwhm_gaussian = 2*wavenum/c * np.sqrt(2*k_B*T*np.log(2)/u)
    fwhm_gaussian = fwhm_gaussian.cgs.value
    fwhm_lorentzian = 2 * gamma
    
    d_ln_wavenum = np.log(all_wavenums[1]/all_wavenums[0])
    if fwhm_gaussian/wavenum < 2 * d_ln_wavenum and fwhm_lorentzian/wavenum < 2 * d_ln_wavenum:
        #print(d_ln_wavenum, fwhm_gaussian/wavenum, fwhm_lorentzian/wavenum)
        index = np.searchsorted(all_wavenums, wavenum)
        cross_sections = np.zeros(len(all_wavenums))
        cross_sections[index] = integral
        return cross_sections
    
    voigt_func = Voigt1D(wavenum, amplitude_L=integral/(np.pi * gamma), fwhm_L=fwhm_lorentzian, fwhm_G = fwhm_gaussian)
    return voigt_func(all_wavenums)
   

def get_Sij(T, wavenums, lower_wavenums, Sij_ref, molecule_id=1, iso_id=1, T_ref=296.0):
    T = float(T)
    
    filename = "QTpy/{}_{}.QTpy".format(molecule_id, iso_id)
    partition_sums = pickle.load(open(filename))
    Q_T = float(partition_sums[str(int(round(T)))])
    Q_ref = float(partition_sums[str(int(T_ref))])
    
    c2 = (h * c / k_B).cgs.value
    ratios = Q_ref/Q_T * np.exp(-c2 * lower_wavenums * (1/T - 1/T_ref)) * (1 - np.exp(-c2*wavenums/T)) / (1 - np.exp(-c2*wavenums/T_ref))
    Sij = Sij_ref * ratios
    print np.min(ratios), np.max(ratios)
    return Sij

np.save("wavenumbers.npy", all_wavenums)

Sij = get_Sij(300, nu, nu_lower, sw)
cross_sections = np.zeros(len(all_wavenums))
for i in range(len(nu)):
    print float(i)/len(nu) * 100 #i, len(nu)
    cross_sections += get_cross_sections(all_wavenums, nu[i], Sij[i], 1e6, 300, gamma_air[i], n_air[i], delta_air[i])    
    
np.save("cross_sections.npy", cross_sections)

