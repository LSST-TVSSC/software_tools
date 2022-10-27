# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:27:23 2018

@author: rstreet
"""

from sys import argv
from os import path, remove
import glob
import json
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import wget

class GaiaSource():
    """Class describing all parameters for a point source detected by Gaia"""

    def __init__(self,params = None):
        self.source_id = None
        self.other_ids = None
        self.ra = None
        self.dec = None
        self.parallax = None
        self.r_est = None
        self.r_lo = None
        self.r_hi = None

        if params != None:

            for key, value in params.items():

                setattr(self,key,value)

    def summary(self):

        output = 'Nearest Gaia source ID='+str(self.source_id)+', RA='+str(self.ra)+', Dec='+str(self.dec)

        if self.parallax < 0.0:
            output += '\nWarning: negative parallax measured by Gaia.  Distance uncertain'

        return output

def analyze_gaia_data():
    """Function to analyze the data returned from a search of the Gaia archive
    around the location of a target of known location, firstly to check
    for a match and if one is found, calculate the measured distance, if
    possible"""

    params = get_args()

    source = search_gaia_source_catalog(params)

    source = check_for_matching_star(params,source)

    source = query_bailer_jones_catalog(params,source)


def search_gaia_source_catalog(params):
    """Function to perform a single object search of the Gaia catalogue,
    using its API service for Table Access Protocol (TAP)
    """

    url = 'http://gaia.ari.uni-heidelberg.de/singlesource/search'

    url = url+'?ra='+params['target_ra']+'&dec='+\
                     params['target_dec']+'&f=json&v=true'

    flist = glob.glob(path.join(params['red_dir'],'SingleSource*'))

    for f in flist:
        remove(f)

    json_file = wget.download(url, out=params['red_dir'])

    data = load_generic_gaia_json(json_file)

    for key, value in data.items():

        if key == 'source':

            source = GaiaSource(params=data[key])

    print('\n'+source.summary())

    return source

def query_bailer_jones_catalog(params,source,verbose=False):
    """Function to query the API for the catalogue of Gaia source positions
    published in Bailer-Jones et al. (2018), AJ, 156, 58"""

    flist = glob.glob(path.join(params['red_dir'],'sync*'))

    for f in flist:
        remove(f)

    url = 'http://gaia.ari.uni-heidelberg.de/tap/sync'

    sql_query = 'REQUEST=doQuery&LANG=ADQL&FORMAT=json&PHASE=RUN&QUERY=SELECT source_id, ra, dec, phot_g_mean_mag, r_est, r_lo, r_hi, teff_val FROM gaiadr2_complements.geometric_distance JOIN gaiadr2.gaia_source USING (source_id) WHERE (source_id='
    sql_query += str(source.source_id)+')'

    url += '?'+sql_query.replace(' ','%20')

    json_file = wget.download(url, out=params['red_dir'])

    data = load_generic_gaia_json(json_file)

    for i,col in enumerate(data['metadata']):

        for key in [ 'r_est', 'r_lo', 'r_hi' ]:

            if col['name'] == key:

                setattr(source, key, data['data'][0][i])

    if verbose:
        print('\nDistance to source: '+str(source.r_est)+' - '+str(source.r_lo)+' + '+str(source.r_hi))

    return source

def calc_distance_from_parallax(gaia_data):
    """Function to calculate the distance to an object based on its parallax"""

    d = 1.0 / abs(gaia_data['parallax']/1000.0)

    sig_d = abs(gaia_data['parallax_error']/gaia_data['parallax']) * d

    gaia_data['distance'] = d
    gaia_data['distance_error'] = sig_d

    return gaia_data

def check_for_matching_star(params,source):
    """Function to compare the target location with those of the returned
    search results to look for a match"""

    target = SkyCoord(params['target_ra'],params['target_dec'],unit=(u.hourangle,u.deg))

    gaia_star = SkyCoord(source.ra,source.dec,unit=(u.deg,u.deg))

    separation = target.separation(gaia_star)

    setattr( source, 'separation', (separation.value * 3600.0) )

    print('Nearest source has a separation of '+str(source.separation)+' arcsec')

    return source

def get_args():
    """Function to read in commandline arguments if available or otherwise
    ask the user for the required parameters"""

    params = {}

    if len(argv) > 1:

        params['target_ra'] = argv[1]
        params['target_dec'] = argv[2]
        params['red_dir'] = argv[3]

    else:

        params['target_ra'] = raw_input('Please enter the target RA in sexigesimal format: ')
        params['target_dec'] = raw_input('Please enter the target Dec in sexigesimal format: ')
        params['red_dir'] = raw_input('Please enter the path to the data directory: ')

    return params

def load_generic_gaia_json(json_file):
    """Function to read in the search results from the API to the Gaia archive"""

    if path.isfile(json_file) == False:
        print('ERROR: Cannot find JSON-format search results from the Gaia archive, file '+json_file)
        exit()

    with open(json_file) as input_file:
        data = json.load(input_file)

    return data

def load_gaia_archive_json(json_file):
    """Function to read in search results from the Gaia archive around
    a given location in JSON format"""

    if path.isfile(json_file) == False:
        print('ERROR: Cannot find JSON-format search results from the Gaia archive, file '+json_file)
        exit()

    with open(json_file) as input_file:
        data = json.load(input_file)

    cols = [ 'source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'phot_g_mean_mag', 'distance', 'distance_error', 'separation' ]
    formats = [ 'S25', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8' ]
    idx = []

    for c in cols:
        for i,item in enumerate(data['metadata']):
            if item['name'] == c:
                idx.append(i)

    table_data = []

    for i,row in enumerate(data['data']):
        row_data = []
        for k,j in enumerate(idx):
            if 'S' in formats[k]:
                row_data.append(data['data'][i][j])
            elif 'f' in formats[k]:
                try:
                    row_data.append(float(data['data'][i][j]))
                except ValueError:
                    row_data.append(None)

        row_data.append(0.0)
        row_data.append(0.0)
        row_data.append(0.0)

        table_data.append( tuple(row_data) )

    t = Table(rows=table_data, names=tuple(cols),
              dtype=tuple(formats))

    return t

if __name__ == '__main__':

    analyze_gaia_data()
