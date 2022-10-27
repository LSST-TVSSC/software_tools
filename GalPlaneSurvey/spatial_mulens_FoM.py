import os
from sys import argv
from sys import path as pythonpath
pythonpath.append('../../')
pythonpath.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rubin_sim.maf as maf
from rubin_sim.data import get_data_dir
from rubin_sim.data import get_baseline
from rubin_sim.utils import hpid2RaDec, equatorialFromGalactic
import healpy as hp
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
import generate_sky_maps
import subprocess
import glob

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
PIXAREA = hp.nside2pixarea(NSIDE,degrees=True)
plot_png = False

SCIENCE_MAPS = ['combined_map', 'galactic_plane_map','magellenic_clouds_map',
                'galactic_bulge_map', 'pencilbeams_map']

class RegionFigureOfMerit:
    def __init__(self):
        self.tE = None
        self.runName = None
        self.i_region = None
        self.gal_lat = None
        self.gal_long = None
        self.n_healpixels = None
        self.detfrac = None
        self.gamma = None
        self.nevents_lsst_max = None
        self.nevents_lsst = None

    def record(self,logfile):
        logfile.write(self.runName+' '+str(round(self.tE,1))+' '+\
                        str(self.i_region)+' '+\
                        str(round(self.gal_lat,2))+' '+\
                        str(round(self.gal_long,2))+' '+\
                        str(round(self.detfrac,2))+' '+\
                        str(round(self.gamma,2))+' '+\
                        str(round(self.nevents_lsst_max,2))+' '+\
                        str(round(self.nevents_lsst,2))+'\n')

    def calc_nevents_lsst(self):
        """Method to estimate the number of events that would be detected
        over the 10yr lifetime of the LSST survey, based on the event rate (Gamma)
        for and the detected fraction of events estimated
        for the region in question.  Note that this gamma is already summed over
        the number of healpixels in the region, to give the event rate for the
        whole region per year.
        """

        if self.gamma and self.detfrac:
            self.nevents_lsst = 10.0 * self.gamma * self.detfrac
            self.nevents_lsst_max = 10.0 * self.gamma
        else:
            self.nevents_lsst = 0
            self.nevents_lsst_max = 0

def calc_spatial_FoM1(region_fom, min_nevents=10.0):
    """Figure of merit calculated over all of the regions of interest,
    calculates the fraction of the regions of interest for which the minimum
    required number of events is reached, over the lifetime of LSST, to allow
    the rate of microlensing events to be measured for each region."""

    FoM = 0.0
    for rFoM in region_fom:
        if rFoM.nevents_lsst >= min_nevents:
            FoM += 1.0
    FoM /= len(region_fom)

    return FoM

def calc_spatial_FoM2(region_fom):
    """Figure of merit calculated over all of the regions of interest,
    calculates the average number of events recovered over the lifetime of LSST
    from regions of interest OUTSIDE THE BULGE (region index 10)."""

    FoM = 0.0
    nr = 0.0
    for rFoM in region_fom:
        if rFoM.i_region != 10:
            FoM += rFoM.nevents_lsst
            nr += 1.0
    FoM /= nr

    return FoM

def calc_spatial_FoM3(region_fom):
    """Figure of merit calculated over all of the regions of interest,
    calculates the median fraction of recovered events, averaged over all
    regions of interest"""

    data = []
    for rFoM in region_fom:
        data.append(rFoM.detfrac)
    FoM = np.median(np.array(data))

    return FoM

def calc_FoM(params):

    # Read in the batch of metric data for a set of different event simulations,
    # and calculate the coadded metric map
    coadd_metric_maps = load_metric_data(params)

    # Load data on pencilbeams from sky map generator since it is easier to
    # identify the pixels from the separate pencilbeam fields from the original
    # configuration than the HEALpixel map data
    pencilbeams = generate_sky_maps.load_optimized_pencilbeams()
    mag_clouds = generate_sky_maps.load_Magellenic_Cloud_data()
    regions_of_interest = pencilbeams + mag_clouds

    for runName,datasets in coadd_metric_maps.items():
        for tEstr, coadd_data in datasets.items():
            (metric_coadd, nevents_per_hp) = coadd_data

            # Log file:
            log_file_path = os.path.join(params['output_dir'],runName+'_microlensing_data.txt')
            if os.path.isfile(log_file_path):
                logfile = open(log_file_path, 'a')
            else:
                logfile = open(log_file_path, 'w')
                logfile.write('# Col 1: runName\n')
                logfile.write('# Col 2: tE [days]\n')
                logfile.write('# Col 3: Region number\n')
                logfile.write('# Col 4: Region Gal lat [deg]\n')
                logfile.write('# Col 5: Region Gal long [deg]\n')
                logfile.write('# Col 6: median detect fraction \n')
                logfile.write('# Col 7: Event rate in pencilbeam area per year \n')
                logfile.write('# Col 8: Expected maximum number of events from LSST\n')
                logfile.write('# Col 9: Expected recovered number of events from LSST\n')

            # Calculate the average recovery fraction for the HEALpixels within each
            # pencilbeam field.
            detfrac_healpix = np.zeros(NPIX)
            ivalid = np.where(metric_coadd > 0.0)[0]
            detfrac_healpix[ivalid] = metric_coadd[ivalid]/nevents_per_hp

            map_FoM = np.zeros(NPIX)
            region_fom = []

            for i,region in enumerate(regions_of_interest):
                rFoM = RegionFigureOfMerit()
                rFoM.runName = runName
                rFoM.tE = float(tEstr)
                rFoM.i_region = i
                rFoM.gal_lat = region.l_center
                rFoM.gal_long = region.b_center
                rFoM.n_healpixels = float(len(region.pixels))
                rFoM.detfrac = np.median(detfrac_healpix[region.pixels])
                rFoM.gamma = lookup_pencilbeam_event_rate(region, PIXAREA)
                rFoM.calc_nevents_lsst()
                rFoM.record(logfile)

                map_FoM[region.pixels] += rFoM.nevents_lsst
                region_fom.append(rFoM)

            # Calculate overall Figure of Merit
            min_nevents = 10.0
            FoM1 = calc_spatial_FoM1(region_fom, min_nevents)
            FoM2 = calc_spatial_FoM2(region_fom)
            FoM3 = calc_spatial_FoM3(region_fom)
            logfile.write('Fraction of pencilbeams with min_events='
                            + str(min_nevents)+' = '+str(round(FoM1,1))+'\n')
            logfile.write('Average number of recovered events outside the Bulge'
                            +' = '+str(round(FoM2,1))+'\n')
            logfile.write('Average fraction of recovered events over all fields '
                            +' = '+str(round(FoM3,1))+'\n')
            logfile.close()

            map_title = 'Number of events detected during LSST , $t_{E}=$' \
                            + tEstr + ', ' + runName
            plot_file = os.path.join(params['output_dir'], 'mulens_event_detected_' \
                                    + tEstr + '_' + runName + '.png')

            plot_FoM(map_FoM, pencilbeams, map_title, plot_file)


def plot_FoM(map, pencilbeams, map_title, plot_file):
    invalid = map == 0.0
    map[invalid] = None

    fig = plt.figure(1,(10,10))
    hp.mollview(map, title=map_title)
    hp.graticule()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close(1)

def load_metric_data(params):
    """Function to load all of the metric data from a set of npz files.
    All files in this directory should be metric data - the data set will be
    co-added.
    """

    file_list = glob.glob(os.path.join(params['data_dir'],'*.npz'))

    coadd_metric_maps = {}
    for file in file_list:
        meta = os.path.basename(file).replace('.npz','').split('_')
        runName = meta[1]+'_'+meta[2]+'_'+meta[3]
        tEstr = meta[-2]
        (metric_coadd, nevents_per_hp) = fetch_coadd_map(coadd_metric_maps,
                                                            runName, tEstr)

        data = np.load(file)
        metric_coadd += data['metric_data']
        nevents_per_hp += 1
        data.close()

        store_coadd_map(coadd_metric_maps, runName, tEstr,
                        metric_coadd, nevents_per_hp)

    print('Coadded metric data maps')

    return coadd_metric_maps

def fetch_coadd_map(coadd_metric_maps, runName, tEstr):

    if runName in coadd_metric_maps.keys():
        tE_array = coadd_metric_maps[runName]
        if tEstr in tE_array.keys():
            (metric_map,nevents_per_hp) = tE_array[tEstr]
        else:
            metric_map = np.zeros(NPIX)
            nevents_per_hp = 0
    else:
        metric_map = np.zeros(NPIX)
        nevents_per_hp = 0

    return metric_map, nevents_per_hp

def store_coadd_map(coadd_metric_maps, runName, tEstr, metric_coadd, nevents_per_hp):
    if runName in coadd_metric_maps.keys():
        tE_array = coadd_metric_maps[runName]
        tE_array[tEstr] = [metric_coadd, nevents_per_hp]
    else:
        tE_array = {tEstr: [metric_coadd, nevents_per_hp]}
    coadd_metric_maps[runName] = tE_array

    return coadd_metric_maps

def lookup_pencilbeam_event_rate(region, PIXAREA):
    """Function to look up the microlensing event rate of a given
    pencilbeam region based on its galactic coordinates, and a data table
    derived from Mroz et al (2020), ApJSS, 249, id.16.

    The returned value of gamma is factored by the area of a HEALpixel for the
    current value of NSIDE, and then by the number of HEALpixels in the given
    region, meaning that it produces the expected number of
    events per year for the whole region.
    """

    # Event rate per year
    # Columns are: gal_long min, max, gal_lat min, max, event rate, event rate per sq deg
    LUT = np.array([
        [190.0, 270.0, -7.0, 7.0, 0.073e-6, 0.017],
        [270.0, 290.0, -7.0, 7.0, 0.379e-6, 0.190],
        [290.0, 300.0, -7.0, 7.0, 0.353e-6, 0.269],
        [300.0, 310.0, -7.0, 7.0, 0.504e-6, 0.482],
        [310.0, 320.0, -7.0, 7.0, 0.919e-6, 0.810],
        [320.0, 330.0, -7.0, 7.0, 0.806e-6, 0.842],
        [330.0, 340.0, -7.0, 7.0, 1.362e-6, 1.461],
        [340.0, 350.0, -7.0, 7.0, 2.122e-6, 2.338],
        [10.0, 20.0, -7.0, 7.0, 3.112e-6, 5.117],
        [20.0, 30.0, -7.0, 7.0, 0.978e-6, 1.360],
        [30.0, 60.0, -7.0, 7.0, 0.889e-6, 0.738]
    ])

    # Set a default event rate to the minimum in the LUT, in case a region
    # is passed that lies outside the given ranges
    default_event_rate = LUT[:,5].min() * PIXAREA * len(region.pixels)

    # Find the LUT entry spatially closest to the given region.
    gamma = default_event_rate
    for i in range(0,len(LUT),1):
        if region.l_center >= LUT[i,0] and region.l_center <= LUT[i,1] \
            and region.b_center >= LUT[i,2] and region.b_center <= LUT[i,3]:
            gamma = LUT[i,5] * PIXAREA * len(region.pixels)

    return gamma


def get_radec_desired_pixels(desired_pixels):

    (theta,phi) = hp.pix2ang(NSIDE,desired_pixels)
    ra_desired_pixels = np.rad2deg(phi)
    dec_desired_pixels = np.rad2deg( (np.pi/2.0) - theta )

    return ra_desired_pixels, dec_desired_pixels

def calc_healpixel_coordinates(healpixels):
    """
    Calculate the l,b of each HEALpixel center from the list of HEALpixels
    """

    pixelPositions = []
    (theta, phi) = hp.pix2ang(NSIDE,range(0,len(healpixels),1))

    ra = np.rad2deg(phi)
    dec = np.rad2deg((np.pi/2.0) - theta)

    skyPositions = SkyCoord(ra*u.deg,
                             dec*u.deg,
                             frame='icrs')
    skyGalactic = skyPositions.transform_to(Galactic)

    return skyGalactic

def get_args():
    params = {}
    if len(argv) == 1:
        params['data_dir'] = input('Please enter the path to directory of metric data files: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')
    else:
        params['data_dir'] = argv[1]
        params['output_dir'] = argv[2]
    return params

if __name__ == '__main__':
    params = get_args()
    calc_FoM(params)
