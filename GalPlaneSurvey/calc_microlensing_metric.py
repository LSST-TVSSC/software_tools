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

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
PIXAREA = hp.nside2pixarea(NSIDE,degrees=True)
plot_png = False

seed = 42
np.random.seed(seed)


def calc_microlensing_metrics(params):
    """Based on a notebook by Peter Yoachim and metric by Natasha Abrams, Markus Hundertmark
    and TVS Microlensing group:
    https://github.com/lsst/rubin_sim_notebooks/blob/main/maf/science/Microlensing%20Metric.ipynb
    """

    bundleList = []

    constraint = 'fiveSigmaDepth > 21.5'
    plotDict = {'colorMax': 950}
    metric1 = maf.mafContrib.microlensingMetric.MicrolensingMetric(metricCalc="detect")

    slicer = HEALpixMicrolensingSlicer(params['t0'],params['u0'],params['tE'],
                                            nside=NSIDE)

    bundleList.append( maf.MetricBundle(metric1, slicer, None, runName=params['runName'],
                                            plotDict=plotDict,
                                            info_label='tE '+str(params['tE'])+' days') )

    # Now we can make the metric bundle, and run the metrics:
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, params['opsim_db'],
                                        outDir='MAFoutput', resultsDb=None)
    bundleGroup.runAll()

    # Extract the values of the metric per HEALpixel as a map:
    outputName = params['runName'].replace('.','_')+\
                '_MicrolensingMetric_detect_tE_'+\
                    str(round(params['tE'],1)).replace('.','_')+\
                        '_days_HEAL'
    metric_data = bundleDict[outputName].metricValues.filled(0.0)

    output_metric_data(params, metric_data)

def output_metric_data(params, metric_data):

    i = 0
    file_path = os.path.join(params['output_dir'], 'mulensMetricData_' \
                + params['runName']+'_'+str(params['t0'])+'_' \
                + str(params['u0'])+'_'+str(params['tE'])+'_'+str(i)+'.npz')
    while os.path.isfile(file_path):
        i += 1
        file_path = os.path.join(params['output_dir'], 'mulensMetricData_' \
                    + params['runName']+'_'+str(params['t0'])+'_' \
                    + str(params['u0'])+'_'+str(params['tE'])+'_'+str(i)+'.npz')

    np.savez_compressed(file_path, metric_data=metric_data)

def HEALpixMicrolensingSlicer(t0,u0,tE,nside=64):
    """
    Generate a UserPointSlicer with a population of microlensing events, suitable
    for use with the MicrolensingMetric.

    This dataSlicer has been adapted from code by Peter Yoachim, with customizations
    to generate a catalog of events for a specific region of the sky rather than
    a uniform random distribution of pointings.

    Parameters
    ----------
    t0 : float
        Time of the closest angular separation of lens-source
    u0 : float
        Impact parameter (ang separation of closest approach) of lens-source
    tE : float
        Einstein crossing time of simulated microlensing event [days]
    nside : int (64)
        HEALpix nside, used to pick which stellar density map to load
    """

    # We simulate a single event for each HEALpixel in the desired survey region,
    # owing to the constraints on how HEALpixel slicers are computed.
    # A set of events is accumulated by repeating the calculation for over the map
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)

    # Add the additional microlensing information about each object to the slicer.
    # In effect this simulates the same event happening at every HEALpix on the sky
    slicer.slicePoints["peak_time"] = np.array( [t0] )
    slicer.slicePoints["crossing_time"] = np.array( [tE] )
    slicer.slicePoints["impact_parameter"] = np.array( [u0] )

    return slicer

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')
        params['t0'] = float(input('Please enter the t0 in MJD-2450000: '))
        params['u0'] = float(input('Please enter the u0: '))
        params['tE'] = float(input('Please enter the tE in days: '))
    else:
        params['opSim_db_file'] = argv[1]
        params['output_dir'] = argv[2]
        params['t0'] = float(argv[3])
        params['u0'] = float(argv[4])
        params['tE'] = float(argv[5])

    params['opsim_db'] = maf.OpsimDatabase(params['opSim_db_file'])
    params['runName'] = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')

    return params

if __name__ == '__main__':
    params = get_args()
    calc_microlensing_metrics(params)
