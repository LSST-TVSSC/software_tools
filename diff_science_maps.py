import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from os import path, mkdir
from sys import argv, exit
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
from astropy.table import Table, Column
from pylab import cm
import csv
import config_utils
import json

# Configuration
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)

def compare_footprint_with_science_maps(params):
    survey_footprint = load_data_table(params['survey_footprint_file'])
    science_footprints = load_data_table(params['science_footprints_file'])
    visibility_map = load_data_table(params['visibility_map_file'])

    output = open(path.join(params['output_dir'],'diff_science_map.dat'),'w')
    output.write('# Science map  Nvisible_pixels  Noverlap_pixels  Ndiff_pixels  %overlap\n')

    for science_map_name in science_footprints.colnames:
        diff_science_map(params, survey_footprint, science_footprints,
                            science_map_name, visibility_map, output)

    output.close()
    
def diff_science_map(params, survey_footprint, science_footprints,
                                science_map_name, visibility_map, output):

    science_map = science_footprints[science_map_name]

    # First, exclude any regions of the science map that lie outside the
    # Rubin visibility zone:
    observable_map = np.zeros(NPIX)
    idx1 = np.where(science_map > 0.0)[0]
    idx2 = np.where(visibility_map['visibility_map'] > 0.0)[0]
    observable_pixels = list(set(idx1).intersection(set(idx2)))
    observable_map[observable_pixels] = 1.0

    # Now identify the set of HEALpixels in both the observable_map and
    # the survey_footprint
    survey_pixels = np.where(survey_footprint['pixelPriority'] > 0.0)[0]
    overlap_pixels = list(set(survey_pixels).intersection(set(observable_pixels)))
    different_pixels = list(set(observable_pixels).difference(set(overlap_pixels)))

    # Calculate the percentage of the map which is covered:
    percent_overlap = (float(len(overlap_pixels))/float(len(observable_pixels))) * 100.0
    difference_map = np.zeros(NPIX)
    observable_map[different_pixels] = -1.0
    output.write(science_map_name+' '+str(len(observable_pixels))+' '
                                     +str(len(overlap_pixels))+' '
                                    +str(len(different_pixels))+' '
                                    +str(percent_overlap)+'\n')

    # Output plot of overlap:
    fig = plt.figure(1,(10,10))
    hp.mollview(observable_map,title='Survey region overlap with '+science_map_name,
                cmap=cm.seismic)
    hp.graticule()
    plt.tight_layout()
    file_path = path.join(params['output_dir'],'diff_map_'+science_map_name+'.png')
    plt.savefig(file_path)
    plt.close(1)

def load_data_table(file_path):
    hdul = fits.open(file_path)
    table_data = hdul[1].data

    table_data = []
    for data_column in hdul[1].columns:
        table_data.append(Column(data=hdul[1].data[data_column.name],
                                 name=data_column.name))

    return Table(table_data)

def get_args():
    params = {}
    if len(argv) == 1:
        params['survey_footprint_file'] = input('Please enter the path to the aggregated survey footprint data table: ')
        params['science_footprints_file'] = input('Please enter the path to the science footprints data table: ')
        params['visibility_map_file'] = input('Please enter the path to the visibility map data table: ')
        params['output_dir'] = input('Please enter the directory path for output: ')
    else:
        params['survey_footprint_file'] = argv[1]
        params['science_footprints_file'] = argv[2]
        params['visibility_map_file'] = argv[3]
        params['output_dir'] = argv[4]

    return params


if __name__ == '__main__':
    params = get_args()
    compare_footprint_with_science_maps(params)
