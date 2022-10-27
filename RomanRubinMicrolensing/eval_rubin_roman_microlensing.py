import os
from sys import argv
from sys import path as pythonpath
pythonpath.append('/Users/rstreet1/software/rubin_sim_gal_plane/rubin_sim/maf/metrics/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rubin_sim.maf as maf
from rubin_sim.data import get_data_dir
from rubin_sim.data import get_baseline
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
import galBulgeRubinRomanMetrics

NSIDE = 64

def run_metrics():

    test = True

    params = get_args()

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Run metrics:
    bundleDict = calc_metrics(opsim_db, runName, test)
    outputName1 = runName.replace('.','_')+'_lensDetectRubinRomanMetric_fiveSigmaDepth_gt_21_5_HEAL'
    outputName2 = runName.replace('.','_')+'_complementaryObsMetric_fiveSigmaDepth_gt_21_5_HEAL'
    if test:
        outputName1 = runName.replace('.','_')+'_lensDetectRubinRomanMetric_fiveSigmaDepth_gt_21_5_USER'
        outputName2 = runName.replace('.','_')+'_complementaryObsMetric_fiveSigmaDepth_gt_21_5_USER'

    metric_data = reshape_metric_data(bundleDict,outputName1,outputName2)

    eval_metrics(metric_data,outputName1,outputName2,runName)

def reshape_metric_data(bundleDict,outputName1,outputName2):

    NPIX = hp.nside2npix(NSIDE)
    metric_data = {}

    # Extract lensDetectRubinRomanMetric values for each HEALpix
    metric_data['lensDetect'] = bundleDict[outputName1].metricValues

    # Extract nContObs,nGapObs values for each HEALpix
    metric_values = bundleDict[outputName2].metricValues
    mapContObs = np.zeros(NPIX)
    mapGapObs = np.zeros(NPIX)
    for i in range(0,len(metric_values),1):
        mapContObs[i] = metric_values[i]['nContObs']
        mapGapObs[i] = metric_values[i]['nGapObs']
    metric_data['nContObs'] = mapContObs
    metric_data['nGapObs'] = mapGapObs

    return metric_data

def eval_metrics(metric_data,outputName1,outputName2,runName):

    #### DOESNT MAKE SENSE TO HEALPIX PLOT SINGLE-NUMBER METRICS

    for metric, data in metric_data.items():
        print(metric, data)
        file_name = runName.replace('.','_')+'_'+metric+'_RomanRubin.png'
        fig = plt.figure(1,(10,10))
        hp.mollview(data)
        hp.graticule()
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(1)

def calc_metrics(opsim_db, runName, test, diagnostics=False):

    bundleList = []
    #metric1 = galBulgeRubinRomanMetrics.lensDetectRubinRomanMetric()
    metric1 = galBulgeRubinRomanMetrics.complementaryObsMetric()

    ### HOW TO PASS JUST THE HEALPIX FOR THE RGES REGION?
    ### OR SUM OVER ALL HEALPIX?
    
    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE)
    if test:
        test_ra = (17.0 + 57.0/60.0 + 34.0/3600.0)*15.0
        test_dec = (29.0 + 13.0/60.0 + 15.0/3600.0)*-1.0
        slicer = maf.UserPointsSlicer(test_ra, test_dec)

    plotDict = {'colorMax': 950}

    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    if diagnostics:
        bundleGroup.plotAll(closefigs=False)

    return bundleDict

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
    else:
        params['opSim_db_file'] = argv[1]

    return params


if __name__ == '__main__':
    run_metrics()
