import os
from sys import argv
from sys import path as pythonpath
pythonpath.append('../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rubin_sim.maf as maf
from rubin_sim.data import get_data_dir
from rubin_sim.data import get_baseline
import healpy as hp
from astropy import units as u
#from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits

# Needs gatspy
from PulsatingStarRecovery import PulsatingStarRecovery_Rachel
import compare_survey_footprints

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
tau_obs = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])
plot_png = False
SCIENCE_MAPS = ['combined_map', 'galactic_plane_map','magellenic_clouds_map',
                'galactic_bulge_map', 'clementini_stellarpops_map',
                'bonito_sfr_map', 'globular_clusters_map', 'open_clusters_map',
                'zucker_sfr_map', 'pencilbeams_map', 'xrb_priority_map']
FILTER_LIST = ['u','g','r','i','z','y']
USE_USER_SLICER = False

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = compare_survey_footprints.load_map_data(params['map_file_path'])

    # Log file:
    logfile = open(os.path.join(params['output_dir'],runName+'_pulsating_star_data.txt'),'w')
    logfile.write('# Col 1: runName\n')
    logfile.write('# Col 2: mapName\n')
    logfile.write('# Col 3: filter\n')
    logfile.write('# Col 4: medianDeltaPabs\n')
    logfile.write('# Col 5: meandeltamag\n')
    logfile.write('# Col 6: meandeltaamp\n')

    # Calculate the metric for the combined map footprint, since this includes
    # all of the other science maps, rather than re-calculate for all maps
    # separately
    mapName = 'combined_map'
    map_data = getattr(map_data_table, mapName)
    desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)
    bundleDict = calc_pulsating_star_metrics(opsim_db, runName, desired_healpix)

    # Extract the metric data from the bundleDict into map format:
    metric_data = extract_metric_data_for_science_map(params,bundleDict,runName,mapName,desired_healpix)

    # Loop over all science survey regions, and calculate the FoM for each region:
    for mapName in SCIENCE_MAPS:
        map_data = getattr(map_data_table, mapName)
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

        FoM = PulsatingStarFiguresOfMerit()
        FoM.runName = runName
        FoM.mapName = mapName

        FoM = eval_metrics(metric_data,desired_healpix,FoM)

        FoM.record(logfile)

    logfile.close()

class PulsatingStarFiguresOfMerit:
    def __init__(self):
        self.tau = None
        self.tau_var = None
        self.runName = None
        self.mapName = None
        self.medianDeltaP = None
        for f in FILTER_LIST:
            setattr(self, 'deltamag_'+f, None)
            setattr(self, 'deltaamp_'+f, None)

    def record(self,logfile):
        for f in FILTER_LIST:
            logfile.write(self.runName+' '+self.mapName+' '+f+' '+\
                    str(self.medianDeltaP)+' '+\
                    str(getattr(self, 'deltamag_'+f))+' '+\
                    str(getattr(self, 'deltaamp_'+f))+'\n')

def get_radec_desired_pixels(desired_pixels):

    (theta,phi) = hp.pix2ang(NSIDE,desired_pixels)
    ra_desired_pixels = np.rad2deg(phi)
    dec_desired_pixels = np.rad2deg( (np.pi/2.0) - theta )

    return ra_desired_pixels, dec_desired_pixels

def HEALpixPulsatingStarSlicer():
    """
    Generate a UserPointSlicer with a population of microlensing events, suitable
    for use with the PulsatingStarRecovery metric.

    This dataSlicer has been adapted from code by Peter Yoachim for a microlensing metric,
    with customizations to generate a catalog of events for a specific region of
    the sky rather than a uniform random distribution of pointings.
    """

    # We simulate a single event for each HEALpixel in the desired survey region,
    # owing to the constraints on how HEALpixel slicers are computed.
    # A set of events is accumulated by repeating the calculation for over the map
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)

    # Add distance information relevant to the PulsatingStarRecovery metric, in MPC
    slicer.slicePoints["distance"] = 0.05

    return slicer


def calc_pulsating_star_metrics(opsim_db, runName, desired_pixels, diagnostics=False):
    """Based on a notebook:
    https://github.com/MARCELLADC/PulsatingStarRecovery/blob/main/Notebook_ANGELO.ipynb
    by Angelo, Macella Di Criscienzo"""

    bundleList = []

    constraint = 'fiveSigmaDepth > 21.5'
    plotDict = {'colorMax': 950}

    # Metric requires input file and default parameter set
    # Configuration taken from original PulsatingStarRecovery repo
    # df contains the list of blending stars with magnitudes; here this is an
    # empty array as we do NOT consider blending
    # Options for the template lightcurve file include ['RRab.csv', 'RRc.csv', 'LPV1.csv']
    # However, cannot iterate over a list because this creates multiple metrics
    # with the same name, so use RRab as a representative example.
    # ERROR in metric line 841: local variable 'noise' referenced before assignment
    pulsating_star_data_path = '../../PulsatingStarRecovery/'
    lc_file = 'RRab.csv'
    distance_modulus = 14.45    # Value based on the Galactic Bulge
    sigmaFORnoise = 1
    do_remove_saturated = True
    numberOfHarmonics = 3
    factorForDimensionGap = 0.75
    df=pd.DataFrame([],columns=[])
    maps = ['DustMap']
    plot_key = False
    metric = PulsatingStarRecovery_Rachel.PulsatingStarRecovery(os.path.join(pulsating_star_data_path,lc_file),
                                                                            sigmaFORnoise,
                                                                            do_remove_saturated,
                                                                            numberOfHarmonics,
                                                                            factorForDimensionGap,
                                                                            df,
                                                                            plot_key)


    if USE_USER_SLICER:
        (ra_desired_pixels, dec_desired_pixels) = get_radec_desired_pixels(desired_pixels)
        slicer = maf.UserPointsSlicer(ra_desired_pixels, dec_desired_pixels, latLonDeg=True)
        slicer.slicePoints["distance"] = 0.05 #Distance in Mpc
    else:
        slicer = HEALpixPulsatingStarSlicer()

    bundleList.append(maf.MetricBundle(metric, slicer, constraint, maps,
                            runName=runName, plotDict=plotDict))

    # Now we can make the metric bundle, and run the metrics:
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    return bundleDict

def extract_metric_data_for_science_map(params,bundleDict,runName,mapName,desired_healpix):
    # Metric produces a list of dictionaries, each containing the set of
    # metrics produced per pixel.
    # Delta_Period_abs is (DeltaP/P)*100 between the input period and
    # that measured from the simulated lightcurves
    if USE_USER_SLICER:
        output_name = runName.replace('.','_')+'_PulsatingStarRecovery_XLynne_blend_fiveSigmaDepth_gt_21_5_USER'
        metric_values = bundleDict[output_name].metricValues
    else:
        output_name = runName.replace('.','_')+'_PulsatingStarRecovery_XLynne_blend_fiveSigmaDepth_gt_21_5_HEAL'
        metric_values = bundleDict[output_name].metricValues.filled(0.0)
    print(bundleDict)
    print(metric_values)

    # Difference in the simulated period versus that fitted from the lightcurves,
    # as a percentage of the original period: diffper_abs=(DeltaP/P)*100
    metric_data = {'Delta_Period_abs': np.zeros(NPIX)}
    for f in FILTER_LIST:
        metric_data['deltamag_'+f] = np.zeros(NPIX)
        metric_data['deltaamp_'+f] = np.zeros(NPIX)

    for i, hp_id in enumerate(desired_healpix):
        #try:
        metric_data['Delta_Period_abs'][hp_id] = metric_values[i]['Delta_Period_abs']
        for f in FILTER_LIST:
            metric_data['deltamag_'+f][hp_id] = metric_values[i]['deltamag_'+f]
            metric_data['deltaamp_'+f][hp_id] = metric_values[i]['deltaamp_'+f]
        #except IndexError:
        #    metric_data['Delta_Period_abs'][hp_id] = np.nan
        #    for f in FILTER_LIST:
        #        metric_data['deltamag_'+f][hp_id] = np.nan
        #        metric_data['deltaamp_'+f][hp_id] = np.nan

    file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_pulsating_star_deltaP.png')
    compare_survey_footprints.plot_map_data(metric_data['Delta_Period_abs'], file_name, range=[0,100.0])

    file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_pulsating_star_deltaP.npz')
    np.savez(file_name, metric_data['Delta_Period_abs'])

    return metric_data

def eval_metrics(metric_data,desired_healpix,FoM):

    FoM.medianDeltaP = np.median(metric_data['Delta_Period_abs'][desired_healpix])

    for f in FILTER_LIST:
        setattr(FoM, 'deltamag_'+f, np.median(metric_data['deltamag_'+f]))
        setattr(FoM, 'deltaamp_'+f, np.median(metric_data['deltaamp_'+f]))

    return FoM

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['map_file_path'] = input('Please enter the path to the GP footprint map: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')
    else:
        params['opSim_db_file'] = argv[1]
        params['map_file_path'] = argv[2]
        params['output_dir'] = argv[3]

    return params


if __name__ == '__main__':
    params = get_args()
    run_metrics(params)
