# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:35:41 2018

@author: rstreet
"""
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun
from astropy.coordinates import get_moon
import copy

def calculate_lsst_field_visibility(fieldRA,fieldDec,start_date,end_date,
                                    min_alt=30.0,dt_days=1.0,
                                    diagnostics=False,verbose=False):
        """Method to calculate the visibility of a given RA and Dec from LSST 
        over the course of a year
        
        Adapted from an example in the Astropy docs.
        
        Inputs:
            :param float fieldRA: Field RA in decimal degrees
            :param float fieldDec: Field Dec in decimal degrees
        """
        
        field = SkyCoord(fieldRA, fieldDec, unit=(u.hourangle, u.deg))
        
        lsst = EarthLocation(lat=-30.239933333333333*u.deg, 
                             lon=-70.7429638888889*u.deg, 
                             height=2663.0*u.m)
        
        total_time_visible = 0.0
        t_start = Time(start_date+' 00:00:00')
        t_end = Time(end_date+' 00:00:00')
        cadence = 0.0007   # In days
        
        n_days = int((t_end - t_start).value)
        
        dates = np.array([t_start + \
                TimeDelta(i,format='jd',scale=None) for i in range(0,n_days,1)])
        
        target_alts = []
        hrs_visible_per_night = []
        hrs_per_night = []
        jds = []
        n_nights_low_vis = 0
        n_nights_hi_vis = 0
        
        f = open('target_visibility_windows.txt','w')
        
        for d in dates:
            
            jds.append(d.jd)
            
            t  = copy.copy(d)
            t.out_subfmt = 'date'
            tstr = t.value
            
            intervals = np.arange(0.0,1.0,cadence)
            
            dt = TimeDelta(intervals,format='jd',scale=None)
            
            ts = d + dt
    
            frame = AltAz(obstime=ts, location=lsst)
            
            altaz = field.transform_to(frame)
            
            alts = np.array((altaz.alt*u.deg).value)
            
            idx = np.where(alts > min_alt)[0]
            
            sun_altaz = get_sun(ts).transform_to(frame)
            
            sun_alts = np.array((sun_altaz.alt*u.deg).value)
            
            jdx = np.where(sun_alts < 12.0)[0]
            
            hrs_per_night.append( cadence*len(sun_alts[jdx])*24.0 )
            
            idx = list(set(idx).intersection(set(jdx)))
            
            target_alts.append(alts[jdx].max())
            
            if len(idx) > 0:
                
                ts_vis = ts[idx]
                
                midyear = Time(str(int(d.decimalyear))+'-07-15T00:00:00.0', format='isot', scale='utc')
                lateyear = Time(str(int(d.decimalyear))+'-11-10T00:00:00.0', format='isot', scale='utc')
                
                tvis = cadence * len(ts_vis)
                
                total_time_visible += tvis
                
                hrs_visible_tonight = tvis*24.0
                
                if hrs_visible_tonight >= 0.5 and hrs_visible_tonight < 4.0:
                    n_nights_low_vis += 1
                    
                elif hrs_visible_tonight >= 4.0:
                    n_nights_hi_vis += 1
                    
                if d < midyear or d > lateyear:
                    f.write(ts_vis.min().value+' '+ts_vis.max().value+' '+str(hrs_visible_tonight)+'\n')
                else:
                    midday = Time(d.datetime.date().strftime("%Y-%m-%d")+'T12:00:00.0', 
                                  format='isot', scale='utc')
                
                    k = np.where(ts_vis < midday)
                    print(k)
                    if len(k) > 0:
                        tmax = ts_vis[k].max()
                        
                        k = np.where(ts_vis > midday)
                        tmin = ts_vis[k].min()
                        
                        f.write(tmin.value+' '+tmax.value+' '+str(hrs_visible_tonight)+'\n')
                
                #target_alts.append(alts[idx].max())
                
                if verbose:
                    print('Target visible from LSST for '+str(round(tvis*24.0,2))+\
                        'hrs on '+tstr)
                        
                hrs_visible_per_night.append((tvis*24.0))
                
            else:
                
                #target_alts.append(-1e5)
                
                hrs_visible_per_night.append(0.0)
                
                if verbose:
                    print('Target not visible from LSST on '+tstr)
        
        f.close()
        
        if diagnostics and len(dates) > 0:
            
            plot_visibility(jds, target_alts, sun_alts, 
                            hrs_visible_per_night, min_alt)
                            
        elif diagnostics and len(dates) == 0:
            
            raise IOError('WARNING: invalid date range input')
        
        print('N nights at low visibility = '+str(n_nights_low_vis))
        print('N nights at hi visibility = '+str(n_nights_hi_vis))
        
        return total_time_visible, hrs_visible_per_night

def plot_visibility(ts, target_alts, sun_alts, 
                    hrs_visible_per_night, min_alt):
    """Function to plot a chart of the target and solar altitude above the 
    horizon at the LSST site as a function of time"""
    
    ts = np.array(ts)
    target_alts = np.array(target_alts)

    (fig, ax1) = plt.subplots(figsize=(10,10))
    
    plt.rcParams.update({'font.size': 18})
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)
    plt.xticks(rotation=45.0)
    
    idx = np.where(target_alts > -1e5)
    ax1.plot((ts-2450000)[idx], target_alts[idx], 'b-', label='Target altitude')
    ax1.set_xlabel('JD')
    ax1.set_ylabel('Maximum altitude [$^{\circ}$]', color='b')
    ax1.xaxis.label.set_fontsize(18)
    ax1.yaxis.label.set_fontsize(18)
    for label in ax1.get_xticklabels():
        label.set_fontsize(18)
    for label in ax1.get_yticklabels():
        label.set_fontsize(18)

    t = [(ts-2450000).min(),(ts-2450000).max()]
    ax1.plot(t,[min_alt]*len(t),'g-.')
    
    ax1.grid(True)
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(ts-2450000, hrs_visible_per_night, 'm--', label='Time target visible')
    ax2.set_ylabel('Hours per night',color='m')
        
    ax2.yaxis.label.set_fontsize(18)
    ax2.grid(False)
    ax2.tick_params('y', colors='m')
    
    fig.tight_layout()
    
    plt.legend()
    
    plt.savefig('target_visibility_from_lsst.png')
    
    plt.close()

if __name__ == '__main__':
    
    if len(argv) > 1:
        
        fieldRA = argv[1]
        fieldDec = argv[2]
        start_date = argv[3]
        end_date = argv[4]
        
    else:
        
        fieldRA = input('Please enter the RA in sexigesimal format, J2000.0: ')
        fieldDec = input('Please enter the Dec in sexigesimal format, J2000.0: ')
        start_date = input('Please enter the start date of the observing window, YYYY-MM-DD: ')
        end_date = input('Please enter the end date of the observing window, YYYY-MM-DD: ')
        
    (total_time_visible,hrs_per_night) = calculate_lsst_field_visibility(fieldRA,fieldDec,
                                                         start_date,end_date,
                                                         diagnostics=True)
    hrs_per_night = np.array(hrs_per_night)
    print('Total time field is visible from Rubin Obs during these dates: '+repr(total_time_visible))
    print('Hours of visibility per night:')
    print(hrs_per_night)
    print('Median hours per night: '+str(np.median(hrs_per_night)))
    print('Total hours observable within window: '+str(hrs_per_night.sum()))