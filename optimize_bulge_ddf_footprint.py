# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:38:32 2018

@author: rstreet
"""
import vizier_tools
import lsst_class
import wfirst_fov
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from matplotlib.patches import Circle
from os import path
from sys import argv
import numpy as np
import bulge_globular_clusters

def optimize_survey_footprint():
    """Function to optimize the DDF survey footprint"""
    
    params = get_args()

    survey_centre = SkyCoord(params['ra']+' '+params['dec'], unit=(u.hourangle, u.deg))

    lsst = lsst_class.LSSTFootprint(ra_centre=params['ra'],
                                    dec_centre=params['dec'])
    
    wfirst_survey = wfirst_fov.bulge_survey_footprint(survey_centre.ra.degree,
                                                      survey_centre.dec.degree)
    
    gcs = bulge_globular_clusters.get_globular_clusters()
    
    gcs = np.array(gcs.values())
    gc_coords = SkyCoord(gcs[:,0],gcs[:,1], unit=(u.deg, u.deg))
    
    lsst.count_stars_in_footprint(gc_coords)
    
    params['search_radius'] = 0.5  # deg
    
    params['lsst_pixscale'] = lsst.pixel_scale
    
    if 'catalog_file' not in params.keys() and '-no-catalog' not in params.keys():

        catalog = fetch_catalog_sources_within_image(params)
    
        catalog.write(path.join(params['red_dir'],'catalog.data'), 
                                format='ascii.basic', overwrite=True)
        
    elif 'catalog_file' in params.keys() and 'no-catalog' not in params.keys():

        catalog = Table.read(params['catalog_file'], format='ascii.basic')
    
    if '-no-catalog' not in params.keys():
        print(catalog)
    
        plot_catalog(params,catalog,lsst,wfirst_survey)
    
    else:
        
        plot_fov(params,lsst,wfirst_survey)
        
def get_args():
    
    params = { }

    if len(argv) > 1:
        params['red_dir'] = argv[1]
        params['ra'] = argv[2]
        params['dec'] = argv[3]
        if len(argv) == 5 and '-catalog=' in argv[4]:
            params['catalog_file'] = str(argv[4]).replace('-catalog=','')
        elif len(argv) == 5 and '-no-catalog' in argv[4]:
            params['-no-catalog'] = True
            
    else:
        print('Parameters:')
        print('> python optimize_bulge_ddf_footprint.py red_dir  ra dec[sexigesimal] [-catalog=file_path or -no-catalog]')
        exit()
    
    return params
    
def fetch_catalog_sources_within_image(params):
    """Function to extract the objects from the VPHAS+ catalogue within the
    field of view of the reference image, based on the metadata information."""
    
    # Radius should be in arcmin
    params['radius'] = params['search_radius']*60.0
    
    catalog = vizier_tools.search_vizier_for_sources(params['ra'], 
                                                       params['dec'], 
                                                        params['radius'], 
                                                        'VPHAS+')
    
    return catalog    

def plot_catalog(params,catalog,lsst,wfirst):
    """Function to plot the field of view"""
    
    c = SkyCoord(params['ra']+' '+params['dec'], unit=(u.hourangle, u.deg))
    
    w = WCS(naxis=2)
    w.wcs.crpix = [0.0, 0.0]
    w.wcs.cdelt = np.array([0.26/3600.0, 0.26/3600.0])
    w.wcs.crval = [c.ra.degree, c.dec.degree]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    
    #plt.subplot(projection=w)    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    plt.subplots_adjust(left=None, bottom=0.2, 
                        right=None, top=None, 
                        wspace=None, hspace=None)
    try:
        plt.scatter(catalog['RAJ2000'], catalog['DEJ2000'], 
                s=(params['lsst_pixscale']/36000.0),
                edgecolor='black', facecolor=(0, 0, 0, 0.2))
    except KeyError:
        plt.scatter(catalog['_RAJ2000'], catalog['_DEJ2000'], 
                s=(params['lsst_pixscale']/36000.0),
                edgecolor='black', facecolor=(0, 0, 0, 0.2))
                    
    for pointing in wfirst:
        
        pointing.draw_footprint(ax)
    
    lsst.draw_footprint(ax)
    
    plt.plot([c.ra.degree],[c.dec.degree],'r+')
    
    #plt.grid(color='white', ls='solid')
    plot_width = 4.0
    xmin = (c.ra.degree - plot_width/2.0)
    xmax = (c.ra.degree + plot_width/2.0)
    ymin = c.dec.degree - plot_width/2.0
    ymax = c.dec.degree + plot_width/2.0
    plt.axis([xmin,xmax,ymin,ymax])
    
    plt.ylabel('Dec [deg]')
    plt.xlabel('RA [deg]')

    plt.xticks(rotation=45.0)
    plt.yticks(rotation=45.0)
    
    plt.savefig(path.join(params['red_dir'],'sky_view.png'), bbox_inches='tight')

def plot_fov(params,lsst,wfirst_survey):
    
    #background = plt.imread('/Users/rstreet/LSST/survey_strategy_wp/PS1_ob180022_4deg2.png')
        
    c = SkyCoord(params['ra']+' '+params['dec'], unit=(u.hourangle, u.deg))
    
    #w = WCS(naxis=2)
    #w.wcs.crpix = [background.shape[1]/2, background.shape[0]/2]
    #w.wcs.cdelt = np.array([4/3600.0, 4/3600.0])
    #w.wcs.crval = [c.ra.degree, c.dec.degree]
    #w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
    fig = plt.figure(1,(10,10))
    
    ax = fig.add_subplot(111, aspect='equal')
    
    plt.subplots_adjust(left=None, bottom=0.2, 
                        right=None, top=None, 
                        wspace=None, hspace=None)
    
    #plt.imshow(background)
    
    for pointing in wfirst_survey:
        
        pointing.draw_footprint(ax)

    lsst.draw_footprint(ax)
    
    plt.plot([c.ra.degree],[c.dec.degree],'r+')
    
    plot_width = 4.0
    xmin = (c.ra.degree - plot_width/2.0)
    xmax = (c.ra.degree + plot_width/2.0)
    ymin = c.dec.degree - plot_width/2.0
    ymax = c.dec.degree + plot_width/2.0
    plt.axis([xmin,xmax,ymin,ymax])
    
    plt.ylabel('Dec')
    plt.xlabel('RA')

    plt.xticks(rotation=45.0)
    plt.yticks(rotation=45.0)

    plt.savefig(path.join(params['red_dir'],'survey_fov.png'), bbox_inches='tight')
    
    plt.close(1)
    
if __name__ == '__main__':
    
    optimize_survey_footprint()
    