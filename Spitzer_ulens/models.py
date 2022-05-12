import numpy as np
import VBBinaryLensing
import os
from MulensModel import Model, SatelliteSkyCoord, MODULE_PATH

#class Models(object):
#    
#    def list_models():
#        models = [func for func in dir(Models) if callable(getattr(Models, func)) 
#                  and not func.startswith("__") 
#                  and not func.startswith("list_")]
#
#class LightCurveModels(Models):
    
def single_lens(time, fb, t0, fs, tE):
    """4-parameters single lens model approximation. Note: need to find the source...
    Args:
        time (1D array): Time array (in days)
        fb (float): Baseline flux.
        t0 (float): time of closest alignment.
        fs (float): 
        tE (float): Einstein radius crossing time
    Return:
        1D array: flux array corresponding to the time stamps.
    """
    ts = (time-t0)/(tE/np.sqrt(12))
    flux = fb+fs/(np.sqrt(ts**2 +1))
    return flux

def single_lens_5(t, tE, t0, fb, fs, u0):
    """5-parameters single lens model approximation (see Gaudi (2011) Review on Exoplanetary Microlensing)
    Args:
        time (1D array): Time array (in days)
        tE (float): Einstein radius crossing time
        t0 (float): Time of closest alignment
        u0 (float): angular separation between lens and source at closest approach in units of Einstein ring.
        fb (float): Baseline flux
        fs (float): Source flux
    Return:
        1D array: flux array corresponding to the time stamps.
    """

    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    flux = fb + A*fs
    return flux

def position_LP(t, alpha, tE, t0, u0):
    """Position of the center of the source with respect to the center of mass.
    Args:
        t (1D array): time array
        alpha (float): angle between lens axis and source trajectory
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
    Returns:
        1D array: tau (time rescale in terms of einstein crossing time)
        1D array: y1 coordinate (along one axis) of source trajectory on the lens plane
        1D array: y2 coordinate (along one axis) of source trajectory on the lens plane
    """
    tau = (t - t0)/tE
    # from Seb. Calchi Novati
    y1 = u0*np.sin(alpha) - tau*np.cos(alpha)
    y2 = -u0*np.cos(alpha) - tau*np.sin(alpha)
    # sign inversed in Fran Bartolic's Example
    #y1 = -u0*np.sin(alpha) + tau*np.cos(alpha)
    #y2 = u0*np.cos(alpha) + tau*np.sin(alpha)
    return tau, y1, y2

def iterate_from(item):
    """Iterating over item
    """
    while item is not None:
        yield item
        item = item.next


def binary_lens(t, s, q, rho, alpha, tE, t0, u0):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol = 1e-03

    # Position of the center of the source with respect to the center of mass.
    tau, y1, y2 = position_LP(t, alpha, tE, t0, u0)

def BL_caustic_curves(VBBL, s, q):
    """Generate caustic and critical curves.
    Args:

    Returns:

    """
    # Calculate the cirtical curves and the caustic curves
    solutions = VBBL.PlotCrit(s, q) # Returns _sols object containing n crit. curves followed by n caustic curves

    # generator function iterating over _sols, _curve, or _point objects 
    # making use of the next keyword
    curves = []
    for curve in iterate_from(solutions.first):
        for point in iterate_from(curve.first):
            curves.append((point.x1, point.x2))

    critical_curves = np.array(curves[:int(len(curves)/2)])
    caustic_curves  = np.array(curves[int(len(curves)/2):])
    return caustic_curves, critical_curves

def binary_lens_mag(t, s, q, rho, alpha, tE, t0, u0, returns='flux'):
    """ Use VBBinaryLensing to generate lightcurve and caustics for a binary lens model.
    Args:
        t (1D array): time array
        s (float): separation between the two lenses in units of total angular Einstein radii
        q (float): mass ratio: mass of the lens on the right divided by mass of the lens on the left
        rho (float): source radius in Einstein radii of the total mass (if not a point-source)
        alpha (float): angle between lens axis and source trajectory
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
    Return: 
        1D array: flux array corresponding to the time stamps.
    Return:
        2D array: x and y coordinates of caustic curves on the lens plane
        2D array: x and y coordinates of critical curves on the lens plane
    """
    # Initialize VBBinaryLensing() class object, set relative accuracy
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol = 1e-03

    # Position of the center of the source with respect to the center of mass.
    tau, y1, y2 = position_LP(t, alpha, tE, t0, u0)

    # empty array where magnification will be stored
    mag = np.zeros(len(tau))

    # calculate the magnification at each time
    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    mag = VBBL.BinaryLightCurve(params, t, y1, y2)
    if returns=='flux':
        return np.array(mag)
    elif returns=='caustics':
        return BL_caustic_curves(VBBL, s, q)

def binary_lens_flux(mag, fb=0, fs=1):
    """Given a magnification curve, a baseline flux and a source flux, this is will return the lightcurve (flux) with appropriate scaling factors.
    Args:
        mag (1D array): magnification curve
        fb (float): baseline flux
        fs (float): source flux
    Return:
        array: flux array with fb and fs scaling factors
    """
    flux = fb + mag*fs
    return flux

def single_lens_flux(mag, fb=0, fs=1):
    """Given a magnification curve, a baseline flux and a source flux, this is will return the lightcurve (flux) with appropriate scaling factors.
    Args:
        mag (1D array): magnification curve
        fb (float): baseline flux
        fs (float): source flux
    Return:
        array: flux array with fb and fs scaling factors
    """
    flux = fb + mag*fs
    return flux

def binary_lens_all(t, s, q, rho, alpha, tE, t0, u0, fb, fs):
    """ Use VBBinaryLensing to generate lightcurve and caustics for a binary lens model.
    Args:
        t (1D array): time array
        s (float): separation between the two lenses in units of total angular Einstein radii
        q (float): mass ratio: mass of the lens on the right divided by mass of the lens on the left
        rho (float): source radius in Einstein radii of the total mass (if not a point-source)
        alpha (float): angle between lens axis and source trajectory
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
    Return: 
        1D array: flux array corresponding to the time stamps.
    Return:
        2D array: x and y coordinates of caustic curves on the lens plane
        2D array: x and y coordinates of critical curves on the lens plane
    """
    # Initialize VBBinaryLensing() class object, set relative accuracy
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol = 1e-03

    # Position of the center of the source with respect to the center of mass.
    tau, y1, y2 = position_LP(t, alpha, tE, t0, u0)

    # empty array where magnification will be stored
    mag = np.zeros(len(tau))

    # calculate the magnification at each time
    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    mag = VBBL.BinaryLightCurve(params, t, y1, y2)

    # get flux light curve
    flux = fb + np.array(mag)*fs
    return flux


def binary_lens_flux_par_gro(t_gro, s, q, rho, alpha, tE, t0, u0, pi_E_N, pi_E_E, fb_gro, fs_gro, coord):
    """Uses Mulens to generate binary lens groud-based lightcurve with microlens parallax.

    """
    # Define model parameters .
    params = {'t_0': t0, 'u_0': u0, 't_E': tE}
    params_pi_E = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
    ra_dec = coord
    params_planet = {'rho': rho, 's': s, 'q': q, 'alpha': -np.rad2deg(alpha)}

    # Set models and satellite settings
    model_planet = Model({**params, **params_planet}, coords = ra_dec)

    # Parallax settings:
    model_planet_parallax = Model({**params, **params_pi_E, **params_planet}, coords = ra_dec) # again for single lens
    model_planet_parallax.parallax(earth_orbital = False, satellite = True)

    # necessary path to ephemeris of the observatory (must for parallax)
    MODULE_PATH = '/Users/ldang/Desktop/spitzer-ulens-tozip-20-01-29/MulensModel'
    satellite = SatelliteSkyCoord(os.path.join(MODULE_PATH, 'data/ephemeris_files', 'Spitzer_ephemeris_02.dat'))

    # defining the binary lens calculation 
    model_planet.set_magnification_methods([np.min(t_gro), 'VBBL', np.max(t_gro)])
    model_planet_parallax.set_magnification_methods([np.min(t_gro), 'VBBL', np.max(t_gro)])

    # get magnification
    mag_gro = model_planet_parallax.magnification(time=t_gro)

    # get flux light curve
    flux_gro = fb_gro + mag_gro*fs_gro
    return flux_gro

def binary_lens_flux_par_sat(t_sat, s, q, rho, alpha, tE, t0, u0, pi_E_N, pi_E_E, fb_sat, fs_sat, coord):
    """Uses Mulens to generate binary lens Spitzer lightcurve with microlens parallax.

    """
    # Define model parameters .
    params = {'t_0': t0, 'u_0': u0, 't_E': tE}
    params_pi_E = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
    ra_dec = coord
    params_planet = {'rho': rho, 's': s, 'q': q, 'alpha': -np.rad2deg(alpha)}

    # Set models and satellite settings
    model_planet = Model({**params, **params_planet}, coords = ra_dec)

    # Parallax settings:
    model_planet_parallax = Model({**params, **params_pi_E, **params_planet}, coords = ra_dec) # again for single lens
    model_planet_parallax.parallax(earth_orbital = False, satellite = True)

    # necessary path to ephemeris of the observatory (must for parallax)
    MODULE_PATH = '/Users/ldang/Desktop/spitzer-ulens-tozip-20-01-29/MulensModel'
    satellite = SatelliteSkyCoord(os.path.join(MODULE_PATH, 'data/ephemeris_files', 'Spitzer_ephemeris_02.dat'))

    # defining the binary lens calculation 
    #model_planet.set_magnification_methods([np.min(t_gro), 'VBBL', np.max(t_gro)])
    model_planet_parallax.set_magnification_methods([np.min(t_sat), 'VBBL', np.max(t_sat)])

    # get magnification
    mag_sat = model_planet_parallax.magnification(t_sat, satellite_skycoord=satellite.get_satellite_coords(t_sat))

    # get flux light curve
    flux_sat = fb_sat + mag_sat*fs_sat
    return flux_sat


def binary_lens_mag_par_both(time_g, time_s, s, q, rho, alpha, tE, t0, u0, pi_E_N, pi_E_E, coord):
    """Uses Mulens to generate Spitzer Parallax lightcurves.
    Args:
        t (1D array): time array
        s (float): separation between the two lenses in units of total angular Einstein radii
        q (float): mass ratio: mass of the lens on the right divided by mass of the lens on the left
        rho (float): source radius in Einstein radii of the total mass (if not a point-source)
        alpha (float): angle between lens axis and source trajectory (in rad)
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
        pi_E_N (float): N-S microlens parallax 
        pi_E_E (float): E-W microlens parallax
        coord (string): target's coordinate in the following format 'hh:mm:ss.ss dd:mm:ss.ss'
    """

    # Define model parameters (PSPL, parallax, target coordinates, planet params)
    params        = {'t_0': t0, 'u_0': u0, 't_E': tE}
    params_pi_E   = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
    ra_dec        = coord
    params_planet = {'rho': rho, 's': s, 'q': q, 'alpha': alpha}

    # Set models and satellite settings .
    model_planet = Model({**params, **params_planet}, coords = ra_dec)

    # Parallax settings:
    model_planet_parallax = Model({**params, **params_pi_E, **params_planet}, coords = ra_dec) # again for single lens
    model_planet_parallax.parallax(earth_orbital = False, satellite = True)

    # necessary path to ephemeris of the observatory (must for parallax)
    MODULE_PATH = '/Users/ldang/Desktop/spitzer-ulens-tozip-20-01-29/MulensModel'
    satellite = SatelliteSkyCoord(os.path.join(MODULE_PATH, 'data/ephemeris_files', 'Spitzer_ephemeris_02.dat'))

    # Calculate finite source magnification using VBBL method for this
    tmin = np.min([np.min(time_g), np.min(time_s)])
    tmax = np.max([np.max(time_g), np.max(time_s)])

    # range of dates :
    model_planet.set_magnification_methods([tmin, 'VBBL', tmax])
    model_planet_parallax.set_magnification_methods ([tmin, 'VBBL', tmax])

    # get magnification curve
    mag_gro = model_planet_parallax.magnification(time=time_g)
    mag_sat = model_planet_parallax.magnification(time=time_s, satellite_skycoord=satellite.get_satellite_coords(time_s))

    return mag_gro, mag_sat

def binary_lens_flux_par_both(time_g, time_s, s, q, rho, alpha, tE, t0, u0, pi_E_N, pi_E_E, fb_g, fs_g, fb_s, fs_s, coord):
    """Uses Mulens to generate Spitzer Parallax lightcurves.
    Args:
        t (1D array): time array
        s (float): separation between the two lenses in units of total angular Einstein radii
        q (float): mass ratio: mass of the lens on the right divided by mass of the lens on the left
        rho (float): source radius in Einstein radii of the total mass (if not a point-source)
        alpha (float): angle between lens axis and source trajectory (in rad)
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
        pi_E_N (float): N-S microlens parallax 
        pi_E_E (float): E-W microlens parallax
        coord (string): target's coordinate in the following format 'hh:mm:ss.ss dd:mm:ss.ss'
    """

    # Define model parameters (PSPL, parallax, target coordinates, planet params)
    params        = {'t_0': t0, 'u_0': u0, 't_E': tE}
    params_pi_E   = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
    ra_dec        = coord
    params_planet = {'rho': rho, 's': s, 'q': q, 'alpha': alpha}

    # Set models and satellite settings .
    model_planet = Model({**params, **params_planet}, coords = ra_dec)

    # Parallax settings:
    model_planet_parallax = Model({**params, **params_pi_E, **params_planet}, coords = ra_dec) # again for single lens
    model_planet_parallax.parallax(earth_orbital = False, satellite = True)

    # necessary path to ephemeris of the observatory (must for parallax)
    MODULE_PATH = '/Users/ldang/Desktop/spitzer-ulens-tozip-20-01-29/MulensModel'
    satellite = SatelliteSkyCoord(os.path.join(MODULE_PATH, 'data/ephemeris_files', 'Spitzer_ephemeris_02.dat'))

    # Calculate finite source magnification using VBBL method for this
    tmin = np.min([np.min(time_g), np.min(time_s)])
    tmax = np.max([np.max(time_g), np.max(time_s)])

    # range of dates :
    model_planet.set_magnification_methods([tmin, 'VBBL', tmax])
    model_planet_parallax.set_magnification_methods ([tmin, 'VBBL', tmax])

    # get magnification curve
    mag_gro = model_planet_parallax.magnification(time=time_g)
    mag_sat = model_planet_parallax.magnification(time=time_s, satellite_skycoord=satellite.get_satellite_coords(time_s))

    # get flux light curve
    flux_gro = fb_g + mag_gro*fs_g
    flux_sat = fb_s + mag_sat*fs_s

    return flux_gro, flux_sat

def single_lens_mag_par_both(time_g, time_s, tE, t0, u0, pi_E_N, pi_E_E, coord):
    """Uses Mulens to generate Spitzer Parallax lightcurves.
    Args:
        t (1D array): time array
        s (float): separation between the two lenses in units of total angular Einstein radii
        q (float): mass ratio: mass of the lens on the right divided by mass of the lens on the left
        rho (float): source radius in Einstein radii of the total mass (if not a point-source)
        alpha (float): angle between lens axis and source trajectory (in rad)
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
        pi_E_N (float): N-S microlens parallax 
        pi_E_E (float): E-W microlens parallax
        coord (string): target's coordinate in the following format 'hh:mm:ss.ss dd:mm:ss.ss'
    """

    # Define model parameters (PSPL, parallax, target coordinates, planet params)
    params        = {'t_0': t0, 'u_0': u0, 't_E': tE}
    params_pi_E   = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
    ra_dec        = coord

    # Set models and satellite settings .
    model = Model({**params}, coords = ra_dec)

    # Parallax settings:
    model_parallax = Model({**params, **params_pi_E}, coords = ra_dec) 
    model_parallax.parallax(earth_orbital = False, satellite = True)

    # necessary path to ephemeris of the observatory (must for parallax)
    MODULE_PATH = '/Users/ldang/Desktop/spitzer-ulens-tozip-20-01-29/MulensModel'
    satellite = SatelliteSkyCoord(os.path.join(MODULE_PATH, 'data/ephemeris_files', 'Spitzer_ephemeris_02.dat'))

    # get magnification curve
    mag_gro = model_parallax.magnification(time=time_g)
    mag_sat = model_parallax.magnification(time=time_s, satellite_skycoord=satellite.get_satellite_coords(time_s))

    return mag_gro, mag_sat

def single_lens_flux_par_both(time_g, time_s, tE, t0, u0, pi_E_N, pi_E_E, fb_g, fs_g, fb_s, fs_s, coord):
    """Uses Mulens to generate Spitzer Parallax lightcurves.
    Args:
        t (1D array): time array
        s (float): separation between the two lenses in units of total angular Einstein radii
        q (float): mass ratio: mass of the lens on the right divided by mass of the lens on the left
        rho (float): source radius in Einstein radii of the total mass (if not a point-source)
        alpha (float): angle between lens axis and source trajectory (in rad)
        tE (float): Einstein radius crossing time
        t0 (float): time of peak magnification
        u0 (float): impact parameter
        pi_E_N (float): N-S microlens parallax 
        pi_E_E (float): E-W microlens parallax
        coord (string): target's coordinate in the following format 'hh:mm:ss.ss dd:mm:ss.ss'
    """

    # Define model parameters (PSPL, parallax, target coordinates, planet params)
    params        = {'t_0': t0, 'u_0': u0, 't_E': tE}
    params_pi_E   = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
    ra_dec        = coord

    # Set models and satellite settings .
    model = Model({**params}, coords = ra_dec)

    # Parallax settings:
    model_parallax = Model({**params, **params_pi_E}, coords = ra_dec) 
    model_parallax.parallax(earth_orbital = False, satellite = True)

    # necessary path to ephemeris of the observatory (must for parallax)
    MODULE_PATH = '/Users/ldang/Desktop/spitzer-ulens-tozip-20-01-29/MulensModel'
    satellite = SatelliteSkyCoord(os.path.join(MODULE_PATH, 'data/ephemeris_files', 'Spitzer_ephemeris_02.dat'))

    # get magnification curve
    mag_gro = model_parallax.magnification(time=time_g)
    mag_sat = model_parallax.magnification(time=time_s, satellite_skycoord=satellite.get_satellite_coords(time_s))

    # get flux light curve
    flux_gro = fb_g + mag_gro*fs_g
    flux_sat = fb_s + mag_sat*fs_s

    return flux_gro, flux_sat

