# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:01:38 2018

@author: rstreet
"""

import calculate_lsst_field_visibility_astropy

def test_calculate_lsst_field_visibility():
    
    ra = '18:20:30.5'
    dec = '-20:45:30.0'
    
    total_time_visible = calculate_lsst_field_visibility_astropy.calculate_annual_lsst_field_visibility(ra,dec,diagnostics=True)
    
    print('Total time visible = '+str(total_time_visible))


if __name__ == '__main__':
    
    test_calculate_lsst_field_visibility()