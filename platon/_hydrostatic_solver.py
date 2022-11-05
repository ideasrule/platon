import cupy as np
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from .errors import AtmosphereError


def _get_radii(ln_Ps, planet_mass, planet_radius,
               P_profile, T_profile, mu_profile):
    intermediate_mu = np.interp(ln_Ps[1:], np.log(P_profile), mu_profile) / 2 + np.interp(ln_Ps[0:-1], np.log(P_profile), mu_profile) / 2
    intermediate_T = np.interp(ln_Ps[1:], np.log(P_profile), T_profile) / 2 + np.interp(ln_Ps[0:-1], np.log(P_profile), T_profile) / 2
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

    R_hill = star_radius * \
        (T_star / T_profile[0])**2 * (planet_mass / (3 * M_sun))**(1.0 / 3)
    max_r_estimate = 1.0 / (1 / planet_radius + k_B * np.median(T_profile) * np.log(
        P_profile[0] / ref_pressure) / (G * planet_mass * np.mean(mu_profile) * AMU))
    # The above equation is negative if the real answer is infinity

    if max_r_estimate < 0 or max_r_estimate > R_hill:
        raise AtmosphereError("Atmosphere unbound: height > hill radius")

    P_below = np.append(ref_pressure, P_profile[P_profile > ref_pressure])
    P_above = np.append(ref_pressure,
                        P_profile[P_profile <= ref_pressure][::-1])
    radii_below = _get_radii(np.log(P_below), planet_mass, planet_radius,
                             P_profile, T_profile, mu_profile)
    radii_above = _get_radii(np.log(P_above), planet_mass, planet_radius,
                             P_profile, T_profile, mu_profile)
    radii = np.append(radii_above.flatten()[1:][::-1],
                      radii_below.flatten()[1:])
    radii = radii[above_cloud_cond]
    dr = -np.diff(radii)

    return radii, dr
