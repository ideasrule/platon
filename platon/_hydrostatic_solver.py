from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import numpy as np
from scipy import integrate
import scipy.interpolate

from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from .errors import AtmosphereError


def _get_radii(ln_Ps, planet_mass, planet_radius,
               T_interpolator, mu_interpolator):
    intermediate_mu = (mu_interpolator(ln_Ps[1:]) + \
                       mu_interpolator(ln_Ps[0:-1])) / 2.0
    intermediate_T = (T_interpolator(ln_Ps[1:]) + \
                      T_interpolator(ln_Ps[0:-1])) / 2.0
    d_inv_r = np.diff(ln_Ps) * k_B * intermediate_T / \
        (G * planet_mass * intermediate_mu * AMU)
    assert(np.all(d_inv_r >= 0) or np.all(d_inv_r <= 0))
    inv_r = 1.0 / planet_radius + np.cumsum(d_inv_r)
    radii = np.append(planet_radius, 1.0 / inv_r)
    return radii


def _solve(P_profile, T_profile, ref_pressure, mu_profile,
           planet_mass, planet_radius, star_radius,
           above_cloud_cond, T_star=None):
    assert(len(P_profile) == len(T_profile))

    # Ensure that the atmosphere is bound by making rough estimates of the
    # Hill radius and atmospheric height
    if T_star is None:
        T_star = Teff_sun

    R_hill = 0.5 * star_radius * \
        (T_star / T_profile[0])**2 * (planet_mass / (3 * M_sun))**(1.0 / 3)
    max_r_estimate = 1.0 / (1 / planet_radius + k_B * np.mean(T_profile) * np.log(
        P_profile[0] / P_profile[-1]) / (G * planet_mass * np.mean(mu_profile) * AMU))
    # The above equation is negative if the real answer is infinity

    if max_r_estimate < 0 or max_r_estimate > R_hill:
        raise AtmosphereError("Atmosphere unbound: height > hill radius")

    mu_interpolator = UnivariateSpline(np.log(P_profile), mu_profile, s=0)
    T_interpolator = UnivariateSpline(np.log(P_profile), T_profile, s=0)

    P_below = np.append(ref_pressure, P_profile[P_profile > ref_pressure])
    P_above = np.append(ref_pressure,
                        P_profile[P_profile <= ref_pressure][::-1])
    radii_below = _get_radii(np.log(P_below), planet_mass, planet_radius,
                             T_interpolator, mu_interpolator)
    radii_above = _get_radii(np.log(P_above), planet_mass, planet_radius,
                             T_interpolator, mu_interpolator)
    radii = np.append(radii_above.flatten()[1:][::-1],
                      radii_below.flatten()[1:])
    radii = radii[above_cloud_cond]
    dr = -np.diff(radii)

    return radii, dr
