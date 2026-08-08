import matplotlib.pyplot as plt

from platon.constants import M_jup, R_jup, R_sun
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator


# All quantities are in SI
Rs = 1.16 * R_sun
Mp = 0.73 * M_jup
Rp = 1.40 * R_jup

cold_profile = Profile()
cold_profile.set_isothermal(900)
cold = TerminatorSector(
    cold_profile,
    cloudtop_pressure=1e3,
    scattering_factor=100,
    scattering_slope=6)

hot_profile = Profile()
hot_profile.set_isothermal(1400)
hot = TerminatorSector(
    hot_profile,
    cloudtop_pressure=1e6,
    scattering_factor=1,
    scattering_slope=4)

terminator = TwoSectorTerminator(cold, hot, cold_fraction=0.5)
calculator = TransitDepthCalculator()
wavelengths, depths, info = calculator.compute_depths(
    terminator, Rs, Mp, Rp, logZ=0, CO_ratio=0.53, full_output=True)

sector_info = info["sectors"]
plt.semilogx(
    1e6 * wavelengths, sector_info["cold"]["unbinned_depths"],
    label="cold sector", alpha=0.7)
plt.semilogx(
    1e6 * wavelengths, sector_info["hot"]["unbinned_depths"],
    label="hot sector", alpha=0.7)
plt.semilogx(1e6 * wavelengths, depths, color="black", label="combined")
plt.xlabel("Wavelength (um)")
plt.ylabel("Transit depth")
plt.legend()
plt.tight_layout()
plt.show()
