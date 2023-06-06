import astropy
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_sun
import numpy as np
import copy

def visibility_over_date_range(target, dates, min_alt=30.0, verbose=False):
    visibility_table = np.zeros(len(dates))
    for t, obs_date in enumerate(dates):
        visibility_table[t] = calculate_visibility(target, obs_date)

    return visibility_table

def calculate_visibility(target, obs_date, min_alt=30.0, verbose=False):
    """Method to calculate the visibility of a given RA and Dec from the Rubin
    Observatory for a specific date.

    Adapted from an example in the Astropy docs.

    Inputs:
        :param SkyCoord target: RA, Dec coordinate of a target, ICRS
        :param Time obs_date: Midnight on the date of observations, assumed to be UTC
    """

    # Location of Rubin Observatory in Cerro Pachon, Chile:
    observatory = EarthLocation(lat=-30.244633*u.deg,
                                lon=-70.749416667*u.deg,
                                height=2647*u.m)

    # Calculate visibility at intervals throughout the night in order to
    # evaluate the number of hours for which the target will be visible
    jd = obs_date.jd
    cadence = 0.04 # days i.e. 1 hr time resolution

    # Store the date formating for logging:
    t  = copy.copy(obs_date)
    t.out_subfmt = 'date'
    tstr = t.value

    # Calculate visibility information for hourly intervals throughout the night
    intervals = np.arange(0.0,1.0,cadence)
    dt = TimeDelta(intervals, format='jd', scale=None)
    ts = obs_date + dt

    # Establish the Earth-based observatory as a frame of reference
    frame = AltAz(obstime=ts, location=observatory)

    # Calculate the target altitude above the local horizon at each timestamp, and
    # identify those times when the target is above the minimum observable altitude
    altaz = target.transform_to(frame)
    alts = np.array((altaz.alt*u.deg).value)
    idx = np.where(alts > min_alt)[0]

    # Calculate the altitude of the Sun from the observatory at each timestamp,
    # and identify when it is below 12.0deg, i.e. the timestamps of local night
    sun_altaz = get_sun(ts).transform_to(frame)
    sun_alts = np.array((sun_altaz.alt*u.deg).value)
    jdx = np.where(sun_alts < 12.0)[0]

    # Store the number of hours of darkness available for this site on this date
    dark_hrs = cadence*len(sun_alts[jdx])*24.0

    # Calculate the intersection of the hours of darkness with target visibility, and
    # the number of hours that the target is visible
    idx = list(set(idx).intersection(set(jdx)))
    #target_alts = alts[jdx].max()

    if len(idx) > 0:

        ts_vis = ts[idx]
        tvis = cadence * len(ts_vis)
        hrs_visible = tvis*24.0

        if verbose:
            print('Target visible for '+str(round(tvis*24.0,2))+\
                'hrs on '+tstr)

    else:

        hrs_visible = 0.0

        if verbose:
            print('Target not visible on '+tstr)

    return hrs_visible
