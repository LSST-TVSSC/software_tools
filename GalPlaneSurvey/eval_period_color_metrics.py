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
from rubin_sim.maf.metrics import periodic_detect_metric, cadence_metrics
from rubin_sim.maf.mafContrib import periodicStarMetric, periodicStarModulationMetric
#from rubin_sim.maf.mafContrib import YoungStellarObjectsMetric
from rubin_sim.maf.mafContrib import microlensingMetric
import compare_survey_footprints

# Needs old lsst.maf import statements updated
#from LSSTunknowns.tdAdnom import filterPairTGapsMetric

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
tau_obs = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])
plot_png = False

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = load_map_data(params['map_file_path'])

    # Log file:
    logfile = open(os.path.join(params['output_dir'],runName+'_period_color_data.txt'),'w')
    logfile.write('# Col 1: runName\n')
    logfile.write('# Col 2: mapName\n')
    logfile.write('# Col 3: tau\n')
    logfile.write('# Col 4: tau_var\n')
    logfile.write('# Col 5: %periodDetect\n')
    logfile.write('# Col 6: medianUniformity\n')
    logfile.write('# Col 7: medianNYSOs\n')
    logfile.write('# Col 8: medianfTGap\n')

    # Compute the metrics for the current map and tau
    bundleDict_map = calc_mapbased_metrics(opsim_db, runName)

    # Loop over all science survey regions:
    for column in map_data_table.columns:
        mapName = column.name
        map_data = getattr(map_data_table, mapName)

        # Determine the HEALpix index of the desired science survey region,
        # taking the Rubin visbility zone into account:
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

        FoM = PeriodColorFiguresOfMerit()
        FoM.runName = runName
        FoM.mapName = mapName

        FoM = eval_spatial_metrics_by_region(params,bundleDict_map,map_data,runName,mapName,
                                     desired_healpix,FoM)

        # Need to loop over all values of tau_obs:
        use_time_metrics = False
        if use_time_metrics:
            for i in range(0,len(tau_obs),1):
                tau = tau_obs[i]
                tau_var = tau_obs[i]*5.0
                FoM.tau = tau
                FoM.tau_var = tau * 5.0

                print('Calculating time-based metrics for tau_var='+str(tau_var))

                bundleDict_time = calc_timebased_metrics(opsim_db, runName, tau_var)

                FoM = eval_time_metrics_by_region(params,bundleDict_time,map_data,runName,mapName,
                                             desired_healpix,tau,FoM)

        FoM.record(logfile)

    logfile.close()

class PeriodColorFiguresOfMerit:
    def __init__(self):
        self.tau = None
        self.tau_var = None
        self.runName = None
        self.mapName = None
        self.percent_periodDetect = None
        self.med_uniformity = None
        self.med_NYSOs = None
        self.med_fTGap = None

    def record(self,logfile):
        logfile.write(self.runName+' '+self.mapName+' '+str(self.tau)+' '+str(self.tau_var)+' '+\
                        str(self.percent_periodDetect)+' '+str(self.med_uniformity)+' '+
                        str(self.med_NYSOs)+' '+str(self.med_fTGap)+'\n')


def eval_spatial_metrics_by_region(params,bundleDict,map_data,runName,mapName,
                            desired_healpix,FoM,datalog=None):
    """Plot spatial maps of the values of both metrics, modulo each of the
    desired spatial regions, and for all four timescale categories.
    Sum the metrics over the desired survey region, and compare with ideal value
    of 1*survey region NHealpix.
    """

    FoM = eval_uniformity(params,runName,mapName,bundleDict,desired_healpix,FoM)
    FoM = eval_YSOMetrics(params,runName,mapName,bundleDict,desired_healpix,FoM)
    FoM = eval_filterPairGaps(params,runName,mapName,bundleDict,desired_healpix,FoM)

    return FoM

def eval_time_metrics_by_region(params,bundleDict,map_data,runName,mapName,
                            desired_healpix,tau,FoM,datalog=None):
    """Plot spatial maps of the values of both metrics, modulo each of the
    desired spatial regions, and for all four timescale categories.
    Sum the metrics over the desired survey region, and compare with ideal value
    of 1*survey region NHealpix.
    """

    FoM = eval_periodDetect(params,runName,mapName,bundleDict,desired_healpix,FoM)
    #FoM = eval_periodicStarMetric(params,runName,mapName,bundleDict,desired_healpix,FoM)

    return FoM

def eval_periodDetect(params,runName,mapName,bundleDict,desired_healpix,FoM):
    """The PeriodicDetectMetric evaluates whether a given HEALpixel receives
    sufficient observations over the length of the survey to measure the period of
    an object with a period of 2d.

    If this is true, it returns a value of 1, if not it returns a value of zero.

    The ideal value of this metric is therefore 1 for each HEALpixel in a desired
    survey region, and the FoM returns the percentage of the desired region
    where this is true.
    """

    output_name = runName.replace('.','_')+'_PeriodicDetectMetric_fiveSigmaDepth_gt_21_5_HEAL'
    metric_data = bundleDict[output_name].metricValues.filled(0.0)

    ideal_sum = float(len(desired_healpix))

    FoM.percent_periodDetect = (metric_data[desired_healpix].sum() / ideal_sum)*100.0

    return FoM

def eval_uniformity(params,runName,mapName,bundleDict,desired_healpix,FoM):

    output_name = runName.replace('.','_')+'_Uniformity_observationStartMJD_fiveSigmaDepth_gt_21_5_HEAL'
    metric_data = bundleDict[output_name].metricValues.filled(0.0)

    FoM.med_uniformity = np.median(metric_data[desired_healpix])

    return FoM

def eval_periodicStarMetric(params,runName,mapName,bundleDict,desired_healpix,FoM):

    output_name = runName.replace('.','_')+'_PeriodicStarMetric_fiveSigmaDepth_gt_21_5_HEAL'
    metric_data = bundleDict[output_name].metricValues.filled(0.0)
    #print(metric_data)

    #FoM.med_uniformity = np.median(metric_data[desired_healpix])

    return FoM

def eval_YSOMetrics(params,runName,mapName,bundleDict,desired_healpix,FoM):

    output_name = runName.replace('.','_')+'_young_stars_fiveSigmaDepth_gt_21_5_HEAL'
    metric_data = bundleDict[output_name].metricValues.filled(0.0)

    FoM.med_NYSOs = np.median(metric_data[desired_healpix])

    return FoM

def eval_filterPairGaps(params,runName,mapName,bundleDict,desired_healpix,FoM):

    output_name = runName.replace('.','_')+'_filterPairTGaps_observationStartMJD_filter_fiveSigmaDepth_fiveSigmaDepth_gt_21_5_HEAL'
    metric_data = bundleDict[output_name].metricValues.filled(0.0)

    # metric_data consists of a dictionary of values for each HEALpix:
    # { 'pixId': pixId, 'Nv': Nv, 'dT_lim': dT_lim,'median': np.median(dT_lim),
    #   'dataSlice': dataSlice }
    # Note that dT_lim is in units of days.
    # We assume that dt_lim < threshold of 1day will be suitable for a
    metric_map = np.zeros(len(metric_data))
    for i in range(0,len(metric_data),1):
        if type(metric_data[i]) == type({}) and not np.isnan(metric_data[i]['median']):
            metric_map[i]= metric_data[i]['median']

    FoM.med_fTGap = np.median(metric_map[desired_healpix])

    return FoM

def load_map_data(map_file_path):
    NSIDE = 64
    NPIX = hp.nside2npix(NSIDE)
    with fits.open(map_file_path) as hdul:
        map_data_table = hdul[1].data

    return map_data_table

def get_priority_threshold(mapName):
    if mapName == 'galactic_plane':
        priority_threshold = 0.4
    elif mapName == 'combined_map':
        priority_threshold = 0.001
    else:
        priority_threshold = 0.0

    return priority_threshold

def calc_mapbased_metrics(opsim_db, runName, diagnostics=False):

    bundleList = []
    metricList = []

    constraint = 'fiveSigmaDepth > 21.5'
    plotDict = {'colorMax': 950}
    #metricList.append(maf.metrics.CountMetric(col=['night'], metricName='Nvis'))

    #
    metricList.append(cadenceMetrics.UniformityMetric())
#    metricList.append(YoungStellarObjectsMetric.NYoungStarsMetric())

    metricList.append(filterPairTGapsMetric.filterPairTGapsMetric(fltpair=['g', 'i'],
                                                        mag_lim=[21, 21],
                                                        dt_lim=[0, 1.5/24],
                                                        save_dT=False,
                                                        allgaps=True))

    # The above metrics can all be run in the standard way using a HEALpixel
    # dataSlicer, so we build these as a loop:
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    for metric in metricList:
        bundleList.append(maf.MetricBundle(metric, slicer, constraint, runName=runName, plotDict=plotDict))

    # Now we can make the metric bundle, and run the metrics:
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    if diagnostics:
        bundleGroup.plotAll(closefigs=False)

    return bundleDict

def calc_timebased_metrics(opsim_db, runName, tau_var, diagnostics=False):

    bundleList = []
    metricList = []

    constraint = 'fiveSigmaDepth > 21.5'
    plotDict = {'colorMax': 950}

    metricList.append(periodicDetectMetric.PeriodicDetectMetric(periods=[tau_var],
                                                                starMags=[20.0],
                                                                amplitudes=[0.1]))

    # PeriodicStarMetric: runs a Monte Carlo simulation of a sinusoidal variable,
    # generates a lightcurve and determines whether or not the period can be
    # measured from the result.
    # Calculates assuming a period of 10d - adjustable?
    # THESE TAKE A LONG TIME TO RUN - being run by Nina Hernikschek?
    # TBD
    #metricList.append(periodicStarMetric.PeriodicStarMetric())
    #metricList.append(periodicStarModulationMetric.PeriodicStarModulationMetric())

    # The above metrics can all be run in the standard way using a HEALpixel
    # dataSlicer, so we build these as a loop:
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    for metric in metricList:
        bundleList.append(maf.MetricBundle(metric, slicer, constraint, runName=runName, plotDict=plotDict))

    # Now we can make the metric bundle, and run the metrics:
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
