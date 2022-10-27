import os
from sys import argv
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
from sys import path as pythonpath
#pythonpath.append('/Users/rstreet1/software/rubin_sim_gal_plane/rubin_sim/maf/metrics/')
from rubin_sim.maf.metrics import galacticPlaneMetrics

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
tau_obs = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0]) # In days
# The tau_obs values corresponding to the categories of NVisits
nvisits_tau_obs = {2.0: 7200,
            5.0: 2880,
            11.0: 1309,
            20.0: 720,
            46.5: 310,
            73.0: 197}
plot_png = False

class FiguresOfMerit:
    def __init__(self):
        self.overlap_healpix = None
        self.overlap_percent = None
        self.missing_healpix = None
        self.missing_percent = None
        self.region_priority_percent = None
        self.nobs_priority_percent = None
        self.footprint_priority = None
        self.ideal_footprint_priority = None

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = load_map_data(params['map_file_path'])
    #mapName = os.path.basename(params['map_file_path'].replace('.fits',''))
    print('Total number of pixels in map: '+str(len(map_data_table)))

    # Assign threshold numbers of visits:
    n_visits_thresholds = calcNVisitsThresholds()

    # Start logging, and loop metric over all science maps:
    logfile = open(os.path.join(params['output_dir'],runName+'_footprint_metric_data.txt'),'w')
    logfile.write('# runName   mapName  tau_obs  NVisits_threshold  Npix_overlap  %overlap  Npix_missing   %missing   footprint_priority  ideal_footprint_priority  %ofPriority  %ofNObsPriority\n')
    for column in map_data_table.columns:
        mapName = column.name

        # Extract the map data for the current science map:
        map_data = getattr(map_data_table, mapName)
        print('Calculating survey region overlap for '+mapName)

        # Calculate the metrics for this science map
        bundleDict = calcNVisits(opsim_db, runName, mapName, diagnostics=False)
        print(bundleDict.keys())

        #test_bundle(bundleDict, mapName, 'STEP 1:')

        # Calculate the Rubin visibility zone:
        #rubin_visibility_zone = calc_rubin_visibility(bundleDict, runName)

        # Determine the HEALpix index of the desired science survey region,
        # taking the Rubin visbility zone into account:
        desired_healpix = calc_desired_survey_map(mapName, map_data)

        #test_bundle(bundleDict, mapName, 'STEP 2:')

        # Loop over each cadence category:
        for i in range(0,len(tau_obs),1):

            # Instantiate FiguresOfMerit object to hold results of analysis:
            FoM = FiguresOfMerit()

            # Calculate the overlap between the OpSim and the desired survey region
            FoM = calcFootprintOverlap(params,runName, mapName,
                                        tau_obs[i], desired_healpix,
                                        bundleDict, FoM)

            #test_bundle(bundleDict, mapName, 'STEP 3:')

            # Calculate the sum priority of all surveyed HEALpixels
            # as a percentage of the ideal values expected from the survey region
            use_metric = True
            if use_metric:
                FoM = eval_footprint_priority(params,map_data, runName, mapName,
                                              tau_obs[i], desired_healpix,
                                              bundleDict, FoM)

            #test_bundle(bundleDict, mapName, 'STEP 4:')

            # Calculate the sum of nObservations and priority of all surveyed HEALpixels
            # as a percentage of the ideal values expected from the survey region
            FoM = eval_footprint_nobs_priority(params,map_data, runName, column.name,
                                                tau_obs[i], desired_healpix,
                                                bundleDict, FoM)

            # Record to the log:
            logfile.write(runName+' '+mapName+' '+str(tau_obs[i])+' '+\
                            str(round(n_visits_thresholds[i],0))+' '+\
                            str(FoM.overlap_healpix)+' '+\
                            str(FoM.overlap_percent)+' '+\
                            str(FoM.missing_healpix)+' '+\
                            repr(FoM.missing_percent)+' '+\
                            repr(FoM.footprint_priority)+' '+\
                            repr(FoM.ideal_footprint_priority)+' '+\
                            repr(FoM.region_priority_percent)+' '+\
                            repr(FoM.nobs_priority_percent)+'\n')

    logfile.close()

def test_bundle(bundleDict, mapName, marker):
    print(marker+': '+mapName)
    outputName = 'GalplaneFootprintMetric_'+mapName+'_NObsPriority'
    metricData = bundleDict[outputName].metricValues.filled(0.0)
    idx = np.argwhere(np.isnan(metricData))
    print('   NaN values: ',idx)
    print('   metricData entries: ',metricData[idx])
    print('   Metric sum values: ',metricData.sum())
    if len(idx) > 0:
        raise IOError('Detected NaNs in metric data')

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['map_file_path'] = input('Please enter the path to the GP footprint map: ')
        params['output_dir'] = input('Please enter the output directory path: ')
    else:
        params['opSim_db_file'] = argv[1]
        params['map_file_path'] = argv[2]
        params['output_dir'] = argv[3]

    return params

def calcFootprintOverlap(params,runName, mapName, tau_obs,
                         desired_healpix, bundleDict, FoM,
                         verbose=False):

    rootName = 'GalplaneFootprintMetric_'+mapName+'_'

    max_possible_npix = float(len(desired_healpix))

    if verbose:
        print('max_possible_npix: ',max_possible_npix)
        print(mapName, len(desired_healpix))
    if max_possible_npix == 0:
        print('ERROR: '+mapName+' has zero overlap with the Rubin visibility zone')
        return FoM

    ## Explore the survey regions covered at least sufficient cadence
    outputName = rootName+'Tau_'+str(tau_obs).replace('.','_')
    metricData = bundleDict[outputName].metricValues.filled(0.0)

    sampled_healpix = np.where(metricData > 0.0)[0]

    overlap_healpix = list(set(sampled_healpix).intersection(set(desired_healpix)))
    FoM.overlap_healpix = len(overlap_healpix)
    unsampled_healpix = list(set(desired_healpix).difference(set(sampled_healpix)))
    missing_healpix = list(set(unsampled_healpix).intersection(set(desired_healpix)))
    FoM.missing_healpix = len(missing_healpix)
    FoM.missing_percent = (float(len(missing_healpix))/max_possible_npix)*100.0
    FoM.overlap_percent = (float(len(overlap_healpix))/max_possible_npix)*100.0
    if verbose:
        print('Number of pixels in overlap region = '+str(len(overlap_healpix))+\
                                                ', '+str(FoM.overlap_percent)+'%')
        print('Number of pixels missing from desired region = '+\
                        str(len(missing_healpix))+', '+str(FoM.missing_percent)+'%')

    # Plot the desired area well sampled in the OpSim footprint
    if plot_png:
        map = np.zeros(NPIX)
        map[overlap_healpix] = 1.0
        file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_'+str(round(tau_obs,0))+'_overlap.png')
        plot_map_data(map, file_name)

    # Plot the desired area missing from the OpSim footprint
    if plot_png:
        map = np.zeros(NPIX)
        map[missing_healpix] = 1.0
        file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_'+str(round(tau_obs,0))+'_missing.png')
        plot_map_data(map, file_name)

    return FoM

def eval_footprint_priority(params,map_data, runName, mapName, tau_obs,
                            desired_healpix, bundleDict, FoM):

    # Rootname of the metric for this map:
    rootName = 'GalplaneFootprintMetric_'+mapName+'_'

    # Calculate the summed HEALpix priority from the whole of the
    # desired survey region, taking visibility region into account
    ideal_footprint = np.zeros(NPIX)
    ideal_footprint[desired_healpix] = map_data[desired_healpix]
    FoM.ideal_footprint_priority = ideal_footprint.sum()

    ## Explore the survey regions covered at least sufficient cadence
    outputName = rootName+'Tau_'+str(tau_obs).replace('.','_')
    metricData = bundleDict[outputName].metricValues.filled(0.0)

    # Using the HEALpix map of the survey footprint which received
    # above the threshold number of observations for this value of tau,
    # sum the combined pixel priority from the observed pixels:
    observed_pixels = np.where(metricData > 0.0)[0]
    FoM.footprint_priority = metricData.sum()
    nObservedPixels = len(observed_pixels)

    # Figure of Merit 1 evaluates the percentage of the pixel priority
    # that received sufficient observations
    FoM.region_priority_percent = (FoM.footprint_priority / FoM.ideal_footprint_priority) * 100.0

    # Plot sky map comparing the summed priority per pixel with the ideal
    #diff_map = -map_data
    #diff_map[observed_pixels] = metric_data[observed_pixels,0]
    if plot_png:
        file_name = os.path.join(params['output_dir'], runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs)+'_obs_footprint_priority.png')
        plot_map_data(metricData, file_name)

    return FoM

def eval_footprint_nobs_priority(params,map_data, runName, mapName, tau_obs,
                            desired_healpix, bundleDict, FoM,
                            verbose=False):

    ### NObsPriority per HEALpix as a function of tau_obs
    outputName = 'GalplaneFootprintMetric_'+mapName+'_NObsPriority'
    metricData = bundleDict[outputName].metricValues.filled(0.0)

    if verbose:
        print(mapName)
        print('METRIC DATA IN SURVEY REGION: ',metricData[desired_healpix])
    # Ideal value of NObsPriority per HEALpix for the current tau_obs,
    # taking into account the Rubin visibility zone and the
    expected_nvisits = nvisits_tau_obs[5.0]
    if verbose:
        print('EXPECTED NVISITS: ',expected_nvisits, tau_obs)
    ideal_metric_data = np.zeros(NPIX)
    ideal_metric_data[desired_healpix] = expected_nvisits * map_data[desired_healpix]
    actual_metric_data = np.zeros(NPIX)
    actual_metric_data[desired_healpix] = metricData[desired_healpix]
    idx = np.argwhere(np.isnan(actual_metric_data))
    if verbose:
        print(idx, actual_metric_data[idx])
        print(map_data[idx])
        print('Actual metric data: ',actual_metric_data[desired_healpix])
        print('Actual metric data range: ',actual_metric_data.min(), actual_metric_data.max())
        print('Ideal metric data: ',ideal_metric_data[desired_healpix])
        print('Actual metric data sum: ',actual_metric_data.sum())
        print('Ideal metric data sum: ',ideal_metric_data.sum())

    # Calculate the difference per HEALpix between the metric values and
    # the ideal values
    percent_metric_data = (actual_metric_data/ideal_metric_data)*100.0
    if plot_png:
        file_name = os.path.join(params['output_dir'], runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs)+'_%nobspriority.png')
        plot_map_data(percent_metric_data, file_name, range=[0.0,100.0])
        file_name = os.path.join(params['output_dir'], runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs)+'_nobspriority.png')
        plot_map_data(actual_metric_data, file_name)
        file_name = os.path.join(params['output_dir'], runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs)+'_ideal_nobspriority.png')
        plot_map_data(ideal_metric_data, file_name)

    # Figure of Merit 2 gives the achieved (nValidObs * HEALpix priority) as
    # a function of the ideal survey value
    FoM.nobs_priority_percent = (actual_metric_data.sum()/ideal_metric_data.sum())*100.0

    return FoM

def calc_rubin_visibility(bundleDict, runName):
    """First iteration of this function extracted HEALpixels with Nvisits
    values > 1.0.  However, this was found to give different results for different
    runs on the same opSim database.   It's unclear why this is the case -
    the number of visits per HEALpix should be an integer, so round up errors
    seem unlikely.
    However, the mask values consistently indicate which HEALpixels receive
    visits and which don't, so we use this to infer the Rubin visibility zone.
    """

    mask = bundleDict[runName.replace('.','_')+\
                '_Nvis_fiveSigmaDepth_gt_21_5_HEAL'].metricValues.mask
    #rubin_visibility_zone = np.where(bundleDict[runName.replace('.','_')+\
    #            '_Nvis_fiveSigmaDepth_gt_21_5_HEAL'].metricValues >= 1.0)[0]
    rubin_visibility_zone = np.where(mask == False)[0]

    #rubin_visibility_map = np.zeros(NPIX)
    #rubin_visibility_map[rubin_visibility_zone] = 1.0

    return rubin_visibility_zone

def calc_desired_survey_map(mapName, map_data):
    if mapName == 'galactic_plane':
        priority_threshold = 0.4
    elif mapName == 'combined_map':
        priority_threshold = 0.001
    else:
        priority_threshold = 0.0
    desired_healpix = np.where(map_data > priority_threshold)[0]

    return desired_healpix

def plot_map_data(map, file_name, range=None):
    fig = plt.figure(1,(10,10))
    if range == None:
        hp.mollview(map)
    else:
        hp.mollview(map, min=range[0], max=range[1])
    hp.graticule()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close(1)

def calcNVisitsThresholds():
    hours_in_obs_season = 6.0 * 30.0 * 24.0
    hours_per_night = 8.0
    n_visits_per_year = (hours_in_obs_season * hours_per_night) / (tau_obs * 24.0)
    n_visits_thresholds = n_visits_per_year * 10.0

    return n_visits_thresholds

def calcNVisits(opsim_db, runName, mapName, diagnostics=False):

    bundleList = []
    metric1 = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
    metric2 = galacticPlaneMetrics.GalPlaneFootprintMetric(science_map=mapName)
    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    plotDict = {'colorMax': 950}
    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric2, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    if diagnostics:
        bundleGroup.plotAll(closefigs=False)

    return bundleDict

def load_map_data(map_file_path):
    NSIDE = 64
    NPIX = hp.nside2npix(NSIDE)
    with fits.open(map_file_path) as hdul:
        map_data_table = hdul[1].data

    return map_data_table


if __name__ == '__main__':
    params = get_args()
    run_metrics(params)
