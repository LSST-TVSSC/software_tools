# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 20:43:52 2018

@author: rstreet
"""

import lsst_visibility_calculator
from novas.compat import make_cat_entry, make_object, make_on_surface
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime

def test_calc_lsst_observing_time():
    
    hrs_visible_per_night = 2.0
    cadence = 0.25
    n_exp_visit = 2
    
    (n_visits, obs_time) = lsst_visibility_calculator.calc_lsst_observing_time(hrs_visible_per_night,
                                                                                cadence,n_exp_visit)
    
    assert n_visits == 7
    assert round(obs_time,4) == 0.1556

def test_setup_LSST_location():
    
    test_site = make_on_surface(10.0, 10.0, 10.0, 10.0, 10.0)
    
    lsst = lsst_visibility_calculator.setup_LSST_location()
    
    assert type(lsst) == type(test_site)
    assert lsst.height == 2663.0
    assert lsst.latitude == -30.244639
    assert lsst.longitude == -70.749417
    
def test_setup_pointing():
    
    pointing = ('Vega', '18:36:56.33635', '+38:47:01.2802')
    
    vega = SkyCoord(pointing[1], pointing[2], unit=(u.hourangle, u.deg))
    
    test_target = make_cat_entry(pointing[0], 'FK5', 0, 
                                           vega.ra.hourangle, vega.dec.deg,
                                           0, 0.0, 0.0, 0.0)
                                           
    target = lsst_visibility_calculator.setup_pointing(pointing)
    
    assert type(target) == type(test_target)

def test_setup_Sun():
    
    test_cat_entry = make_cat_entry('DUMMY', 'xxx', 0, 0.0, 0.0, 
                                       0.0, 0.0, 0.0, 0.0)
    
    test_sun = make_object(0, 10, 'Sun', test_cat_entry)
    
    sun = lsst_visibility_calculator.setup_Sun()
    
    assert type(sun) == type(test_sun)

def test_parse_window_dates():
    
    date1 = '2018-09-12'
    date2 = '2018-09-15'
    
    test_date = datetime.utcnow()
    
    params = lsst_visibility_calculator.parse_window_dates(date1, date2)
    
    assert type(params[0]) == type(0.0)     # jd_start
    assert params[0] > 2400000.0
    assert type(params[1]) == type(0.0)     # jd_end
    assert params[1] > 2400000.0
    assert type(params[2]) == type(test_date)   # start_date
    assert type(params[3]) == type(test_date)   # end_date
    assert type(params[4]) == type(0.0)         # delta_t
    assert abs(params[4]) < 100.0
    assert type(params[5]) == type(0.0)           # n_leap_secs
    