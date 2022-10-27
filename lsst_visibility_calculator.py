# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:36:34 2018

@author: Rachel Street based on code by Markus Hundertmark
"""

from sys import argv
from novas.compat import make_on_surface, julian_date, app_planet
from novas.compat import make_cat_entry, make_object, topo_star, equ2hor
from novas.compat.eph_manager import ephem_open
ephem_open('JPLEPH')
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def calc_hours_visibility_from_LSST(pointing, start_date, end_date, 
                                    n_exp_visit, cadence, exp_time):
    """Function to calculate the number of hours a given pointing is visible
    from LSST between specified dates.
    
    :param tuple pointing: Field name, field center RA, Dec J2000.0, sexigesimal
    :param string start_date: Start of visibility window, YYYY-MM-DD
    :param string end_date: End of visibility window, YYYY-MM-DD
    """
    
    accuracy = 0        # Full accuracy = 0, reduced accuracy = 1
    error = 0
    
    # Coordinates of the CIO with respect to the ITRS pole for 2008 April 24 
    # to be revised!!!!
    x_pole = -0.002
    y_pole = +0.529

    lsst_site = setup_LSST_location()
    
    (jd_start, jd_end, start_date, end_date, delta_t, n_leap_secs) = parse_window_dates(start_date, 
                                                                                        end_date)
    
    target = setup_pointing(pointing)
    
    sun = setup_Sun()
    
    jd_dates = np.arange(jd_start, jd_end, 1.0)      # Per day
    
    n_days = (end_date - start_date).days
    
    dates = np.array([start_date + timedelta(days=i) for i in range(0,n_days,1)])
    
    target_alts = []
    sun_alts = []
    hrs_visible_per_night = []
    hrs_per_night = []
    n_visits_per_night = []
    obs_time_per_night = []
    
    for i in range(0,len(jd_dates),1):
        
        jd = jd_dates[i]
        
        tt = jd + (n_leap_secs + 32.184) / 86400.0
        
        (zd_target, zd_sun) = calc_target_and_sun_zenith_distance(jd, tt, delta_t, 
                                                            lsst_site, 
                                                            target, sun)
        
        #target_alts.append(zd_target-90.0)
        sun_alts.append(zd_sun-90.0)
        
        mins_night = 0.0
        mins_visible = 0.0
        peak_alt = -1e5
        
        for dt in np.arange(0.0,1.0,0.0007):
        
            (zd_target, zd_sun) = calc_target_and_sun_zenith_distance(jd+dt, tt+dt, 
                                                                delta_t, 
                                                                lsst_site, 
                                                                target, sun)
            
            if zd_sun >= 102.0:
                
                mins_night += 1.0
                
                if (90-zd_target) > peak_alt:
                    peak_alt = 90-zd_target
                    
                if zd_target < 70.0:
                    mins_visible += 1.0
                    
        target_alts.append(peak_alt)
        
        hrs_per_night.append(mins_night/60.0)
        
        if mins_visible == 0.0:
            print('Target not visible on '+dates[i].strftime("%Y-%m-%d")+\
                                                            ' ('+str(jd)+') '+\
                                                            'length of night '+\
                                                            str(round(hrs_per_night[-1],2))+'hrs')
            hrs_visible_per_night.append(0.0)
            
        else:
            print('Target visible for '+str(round(mins_visible/60.0,2))+\
                    'hrs on '+dates[i].strftime("%Y-%m-%d")+' ('+str(jd)+') '+\
                                                            'length of night '+\
                                                            str(round(hrs_per_night[-1],2))+'hrs')
                                                            
            hrs_visible_per_night.append( mins_visible/60.0 )
        
        (n_visits, obs_time) = calc_lsst_observing_time(hrs_visible_per_night[-1],
                                                        cadence,n_exp_visit,
                                                        exp_time)
        
        n_visits_per_night.append(n_visits)
        obs_time_per_night.append(obs_time)
        
        
    target_alts = np.array(target_alts)
    sun_alts = np.array(sun_alts)
    hrs_visible_per_night = np.array(hrs_visible_per_night)
    n_visits_per_night = np.array(n_visits_per_night)
    obs_time_per_night = np.array(obs_time_per_night)
    hrs_per_night = np.array(hrs_per_night)
    
    plot_visibility(pointing, jd_dates, dates, target_alts, sun_alts, 
                    hrs_visible_per_night, n_visits_per_night, obs_time_per_night)
    
    total_visibility = hrs_visible_per_night.sum()
    total_obs_time = obs_time_per_night.sum()
    total_night_time = hrs_per_night.sum()
    
    print('\n')
    print('Total target visibility over window = '+str(round(total_visibility,2))+'hrs')
    print('Total open-shutter time required = '+str(round(total_obs_time,2))+'hrs')
    print('Total nighttime hours over window = '+str(round(total_night_time))+'hrs')
    
def calc_target_and_sun_zenith_distance(jd, tt, delta_t, lsst_site, target,sun):
    
    accuracy = 0        # Full accuracy = 0, reduced accuracy = 1
    error = 0
    
    # Coordinates of the CIO with respect to the ITRS pole for 2008 April 24 
    # to be revised!!!!
    x_pole = -0.002
    y_pole = +0.529

    (topo_ra, topo_dec) = topo_star(tt, delta_t, target, lsst_site, accuracy)
        
    (zd_target,az_target),(rar,decr) = equ2hor(jd, delta_t, x_pole, y_pole, 
                                                lsst_site, 
                                                topo_ra, topo_dec, 0, 0)
    
    (topo_sun_ra, topo_sun_dec, sun_dist) = app_planet(tt, sun)
    
    (zd_sun,az_sun),(rar1,decr1) = equ2hor(jd, delta_t, x_pole, y_pole, 
                                            lsst_site, 
                                            topo_sun_ra, topo_sun_dec, 1, 0)
    
    return zd_target, zd_sun
    
def setup_LSST_location():
    """Function to build a NOVAS-compatible observatory descriptor for
    the LSST site"""
    
    height = 2663.0
    latitude = -30.244639
    longitude = -70.749417
    
    temperature = 10.0
    pressure = 750.0

    lsst = make_on_surface(latitude, longitude, height, 
                                        temperature, pressure)
    
    return lsst

def parse_window_dates(start_date_str, end_date_str):
    """Function to convert the observing window start and end dates from strings
    in YYYY-MM-DD format to JD as required by NOVAS and calcuate the 
    leap second offset needed for later calculations.
    """
    
    def calc_JD_UTC(date_string):
        
        (year,month,day) = date_string.split('-')
        hour = 0.0
        
        jd_utc = julian_date(int(year), int(month), int(day), hour)
        
        return jd_utc
        
    n_leap_secs = 33.0
    
    ut1_utc = -0.387845 # Difference between UT1 and UTC
    
    jd_start = calc_JD_UTC(start_date_str)
    jd_end = calc_JD_UTC(end_date_str)
    
    jd_ut1 = jd_start + ut1_utc / 86400.0
    delta_t = 32.184 + n_leap_secs - ut1_utc
    
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d")
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d")
    
    return jd_start, jd_end, start_date, end_date, delta_t, n_leap_secs

def setup_pointing(pointing):
    """Function to establish the field center pointing as a NOVAS-format
    celestial catalog entry.
    
    Proper motion, parallax and radial motion are not yet supported.
    """
    
    s = SkyCoord(pointing[1], pointing[2], unit=(u.hourangle, u.deg))
    
    target = make_cat_entry(pointing[0], 'FK5', 0, 
                                           s.ra.hourangle, s.dec.deg,
                                           0, 0.0, 0.0, 0.0)
    
    # In NOVAS notation, (2,0) indicates a star. (0,10) is the Sun
    #target = make_object(2, 0, pointing[0], field)
    
    return target

def setup_Sun():
    """
    Function to establish a NOVAS-object for the Sun
    """
    
    star = make_cat_entry('DUMMY', 'xxx', 0, 0.0, 0.0, 
                                       0.0, 0.0, 0.0, 0.0)
    
    sun = make_object(0, 10, 'Sun', star)
    
    return sun
    
def plot_visibility(pointing, jd_dates, dates, target_alts, sun_alts, 
                    hrs_visible_per_night, n_visits_per_night, obs_time_per_night):
    """Function to plot a chart of the target and solar altitude above the 
    horizon at the LSST site as a function of time"""
    
    (fig, ax1) = plt.subplots(figsize=(10,10))
    
    plt.rcParams.update({'font.size': 18})
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)
    plt.xticks(rotation=45.0)
    
    ax1.plot(dates, target_alts, 'b-', label='Target altitude')
    #ax1.plot(jd_dates-2450000, sun_alts, 'k-.', label='Sun')
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Maximum altitude [$^{\circ}$]', color='b')
    ax1.set_title('Visibility of '+pointing[0]+' from LSST')
    ax1.xaxis.label.set_fontsize(18)
    ax1.yaxis.label.set_fontsize(18)
    for label in ax1.get_xticklabels():
        label.set_fontsize(18)
    for label in ax1.get_yticklabels():
        label.set_fontsize(18)
        
    ax2 = ax1.twinx()
    ax2.plot(dates, hrs_visible_per_night, 'm--', label='Time target visible')
    ax2.set_ylabel('Hours per night',color='m')
    
    ax2.plot(dates, obs_time_per_night, 'g-.', label='Open-shutter time')
    
    ax2.yaxis.label.set_fontsize(18)
    
    fig.tight_layout()
    
    plt.legend()
    
    plt.savefig('visibility_'+str(pointing[0]).replace(' ','_')+'_lsst.png')
    
    plt.close()

def calc_lsst_observing_time(hrs_visible_per_night,cadence,n_exp_visit,exp_time):
    """Function to calculate how much observing time would be used in a given
    night (including overheads), for a given cadence of re-visiting the 
    pointing provided.
    
    :param float hrs_visible_per_night: Total number of hours visibility for a
                                        given night
    :param float cadence: Interval between repeated visits in decimal hours
    :param int n_exp_visit: Number of exposures per visit
    
    Exposures are assumed to have the LSST standard length of 15s.
    A single visit in assumed to consist of exposures in the same filter, but
    the filter change time is factored in to allow for the maximum possible
    overhead.
    """
    
    # Overheads quoted in seconds
    readout = 0.0 # -> 2s, but included in slew
    slew = 12.0  
    filter_change = 120.0 
    
    # Time required for a single visit [secs]:
    visit_time = slew + filter_change + n_exp_visit * (exp_time + readout)
    visit_time = visit_time/3600.0
    
    # Time between repeated visits to the target within the same night    
    repeat_cycle = visit_time + cadence
    
    if hrs_visible_per_night < visit_time:
        n_visits = 0

    else:
        n_visits = int(hrs_visible_per_night / repeat_cycle)
    
    obs_time = n_visits * visit_time
    
    return n_visits, obs_time
    
if __name__ == '__main__':
    
    if len(argv) == 1:
        
        ra = input('Please enter the RA of your target [sexigesimal]: ')
        dec = input('Please enter the Dec of your target [sexigesimal]: ')
        name = input('Please enter the name of your target: ')
        date1 = input('Please enter the start of the visibility window [YYYY-MM-DD]: ')
        date2 = input('Please enter the end of the visibility window [YYYY-MM-DD]: ')
        n_exp_visit = int(input('Please enter the number of exposures per visit [int]: '))
        cadence = float(input('Please enter the interval between visits to target [hrs]: '))
        exp_time = float(input('Please enter the exposure time in seconds: '))
        
    else:
        
        ra = argv[1]
        dec = argv[2]
        name = argv[3]
        date1 = argv[4]
        date2 = argv[5]
        n_exp_visit = int(argv[6])
        cadence = float(argv[7])
        exp_time = float(argv[8])
        
    pointing = (name, ra, dec)
    
    calc_hours_visibility_from_LSST(pointing, date1, date2, n_exp_visit, 
                                    cadence, exp_time)
    