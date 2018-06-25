from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import numpy as np
from scipy import integrate
import scipy.interpolate

from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from .errors import AtmosphereError

def _test_solve(ln_Ps, planet_mass, planet_radius,
                T_interpolator, mu_interpolator):
    #avg_mu = (mu[1:] + mu[0:-1])/2.0
    intermediate_mu = (mu_interpolator(ln_Ps[1:]) + mu_interpolator(ln_Ps[0:-1]))/2.0
    intermediate_T = (T_interpolator(ln_Ps[1:]) + T_interpolator(ln_Ps[0:-1]))/2.0

    d_inv_r = np.diff(ln_Ps) * k_B * intermediate_T/(G*planet_mass*intermediate_mu*AMU)

    assert(np.all(d_inv_r >= 0) or np.all(d_inv_r <= 0))
    inv_r = 1.0/planet_radius + np.cumsum(d_inv_r)
    radii = np.append(planet_radius, 1.0/inv_r)
    return radii - planet_radius

def _isothermal_solve(temperature, P_profile, ref_pressure, planet_mass, planet_radius, mean_mu, above_cloud_cond):
    radii = 1.0/(1/planet_radius + k_B*temperature*np.log(P_profile/ref_pressure)/(G * planet_mass * mean_mu * AMU))
    radii = radii[above_cloud_cond]
    dr = -np.diff(radii)
    return radii, dr
    

def _ode_solve(P_profile, T_profile, ref_pressure, mu, planet_mass,
               planet_radius, star_radius, above_cloud_cond,
               T_star=None, approximate=True):
    assert(len(P_profile) == len(T_profile))
    mu_interpolator = UnivariateSpline(np.log(P_profile), mu, s=0)
    T_interpolator = UnivariateSpline(np.log(P_profile), T_profile, s=0)

    # Solve the hydrostatic equation
    def hydrostatic(y, lnP):
        r = y + planet_radius
        T_local = T_interpolator(lnP)
        local_mu = mu_interpolator(lnP)
        dy_dlnP = -r**2 * k_B * T_local/(G * planet_mass * local_mu * AMU)
        return dy_dlnP

    P_below = np.append(ref_pressure, P_profile[P_profile > ref_pressure])
    P_above = np.append(ref_pressure, P_profile[P_profile <= ref_pressure][::-1])        
    if approximate:
        heights_below = _test_solve(np.log(P_below), planet_mass, planet_radius, T_interpolator, mu_interpolator)
        heights_above = _test_solve(np.log(P_above), planet_mass, planet_radius, T_interpolator, mu_interpolator)        
    else:
        print("Using exact solver")
        heights_below, infodict = integrate.odeint(
            hydrostatic, 0, np.log(P_below), full_output=True)
        heights_below = heights_below.flatten()
        if infodict["message"] != "Integration successful.":
            raise AtmosphereError("Hydrostatic solver failed")

        heights_above, infodict = integrate.odeint(
            hydrostatic, 0, np.log(P_above), full_output=True)
        heights_above = heights_above.flatten()
        if infodict["message"] != "Integration successful.":
            raise AtmosphereError("Hydrostatic solver failed")    
    
    radii = planet_radius + np.append(heights_above.flatten()[1:][::-1],
                      heights_below.flatten()[1:])
    #np.save("approximate.npy", radii)
    radii = radii[above_cloud_cond]
    dr = -np.diff(radii)
    return radii, dr

def _solve(P_profile, T_profile, ref_pressure, mu_profile,
           planet_mass, planet_radius, star_radius,
           above_cloud_cond, T_star=None, approximate=True):
    assert(len(P_profile) == len(T_profile))

    # Ensure that the atmosphere is bound by making rough estimates of the
    # Hill radius and atmospheric height
    if T_star is None:
        T_star = Teff_sun

    R_hill = 0.5*star_radius*(T_star/T_profile[0])**2 * (planet_mass/(3*M_sun))**(1.0/3)
    max_r_estimate = 1.0/(1/planet_radius + k_B*np.mean(T_profile)*np.log(P_profile[0]/P_profile[-1])/(G * planet_mass * np.mean(mu_profile) * AMU))
    # The above equation is negative if the real answer is infinity

    if max_r_estimate < 0 or max_r_estimate > R_hill:
        raise AtmosphereError("Atmosphere unbound: height > hill radius")
        
    #if len(np.unique(T_profile)) == 1 and approximate:
    #    return _isothermal_solve(T_profile[0], P_profile, ref_pressure, planet_mass, planet_radius, np.mean(mu_profile), above_cloud_cond)

    #isothermal_radii, isothermal_dr = _isothermal_solve(T_profile[0], P_profile, ref_pressure, planet_mass, planet_radius, np.mean(mu_profile), above_cloud_cond)

    radii, dr = _ode_solve(P_profile, T_profile, ref_pressure, mu_profile,
                      planet_mass, planet_radius, star_radius,
                           above_cloud_cond, T_star=T_star, approximate=approximate)
    #diff = np.abs(radii - isothermal_radii)/radii
    #print("Max diff", np.max(diff))
    return radii, dr
