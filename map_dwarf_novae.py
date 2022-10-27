# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:37:16 2018

@author: rstreet
"""

from astropy.table import Table, Column
from os import path
from sys import argv
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import lsst_class

def map_dwarf_novae(catalog_file,ra_cen,dec_cen):
    """Function to map the dwarf novae discovered by OGLE and identify how
    many of them should fall within a given LSST pointing"""
    
    DN = read_dwarf_novae_catalog(catalog_file)
    
    lsst = LSSTFootprint(ra_centre=ra_cen, dec_centre=dec_cen)

    coords = SkyCoord(ra=DN['ra']*u.degree, 
                      dec=DN['dec']*u.degree, 
                      frame='icrs')
    
    lsst.count_stars_in_footprint(coords)
    
def read_dwarf_novae_catalog(catalog_file):
    """Function to read in the catalog of dwarf novae from 
    https://ui.adsabs.harvard.edu/?#abs/2015AcA....65..313M
    by Mroz, P et al.
    """
    
    if not path.isfile(catalog_file):
        raise IOError('Cannot find input catalog file '+catalog_file)
        exit()

    lines = open(catalog_file,'r').readlines()
    
    ra = []
    dec = []
    name = []
    field = []
    imag = []
    iamp = []
    freq = []
    dur = []
    
    for l in lines:
        if '#' not in l[0:1]:
            entries = l.replace('\n','').split()
            ra.append(float(entries[0]))
            dec.append(float(entries[1]))
            name.append(entries[2])
            field.append(entries[9])
            imag.append(float(entries[11]))
            iamp.append(float(entries[12]))
            freq.append(float(entries[13]))
            dur.append(float(entries[14]))
    
    
    DN = Table()
    DN['ra'] = ra
    DN['dec'] = dec
    DN['name'] = name
    DN['field'] = field
    DN['imag'] = imag
    DN['iamp[mag]'] = iamp
    DN['frequency[yr-1]'] = freq
    DN['duration[d]'] = dur
    
    print('Read catalogue data')
    
    return DN
    
if __name__ == '__main__':

    if len(argv) == 1:
        catalog_file = input('Please enter the path to the catalogue file: ')
        ra_cen = input('Please enter the LSST field centre RA [sexigesimal]: ')
        dec_cen = input('Please enter the LSST field centre Dec [sexigesimal]: ')
    else:
        catalog_file = argv[1]
        ra_cen = argv[2]
        dec_cen = argv[3]
        
    map_dwarf_novae(catalog_file,ra_cen,dec_cen)