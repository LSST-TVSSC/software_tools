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
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
#from sys import path as pythonpath
#pythonpath.append('/Users/rstreet1/software/rubin_sim_gal_plane/rubin_sim/maf/metrics/')
from rubin_sim.maf.metrics import galacticPlaneMetrics
from compare_survey_footprints import calcNVisitsThresholds

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

def eval_footprint_priority():

    print('''THIS CODE IS DEPRECIATED AND MAINTAINED AS A RECORD.
            PLEASE USE compare_survey_footprints.py INSTEAD''')
    exit()
    
    params = get_args()

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = load_map_data(params['map_file_path'])
    mapName = os.path.basename(params['map_file_path'].replace('.fits',''))
    print('Total number of pixels in map: '+str(len(map_data_table)))

    # Assign threshold numbers of visits:
    n_visits_thresholds = calcNVisitsThresholds()

    # Start logging, and loop metric over all science maps:
    logfile = open('footprint_metric_data.txt','w')
    logfile.write('runName   mapName  tau_obs  NVisits_threshold  nObserved_pixels footprint_priority  ideal_footprint_priority  %ofPriority  %ofNObsPriority\n')
    for column in map_data_table.columns:
        # Calculate the total numbers of visits per HEALpix for this strategy
        bundleDict = calcMetric(opsim_db, runName, column.name, diagnostics=False)

        map_data = getattr(map_data_table, column.name)
        print('Calculating survey region overlap for '+column.name)

        #metric_data = unpackMetricData(runName, bundleDict)

        evalFootprintPriority(opsim_db, map_data, runName, column.name,
                                n_visits_thresholds, bundleDict, logfile)
    logfile.close()

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['map_file_path'] = input('Please enter the path to the footprint map summed over all filters: ')
    else:
        params['opSim_db_file'] = argv[1]
        params['map_file_path'] = argv[2]

    return params

def unpackMetricData(runName, bundleDict):

    rootName = 'GalplaneFootprintMetric_combined_map_'
    metric_data = np.zeros((NPIX,2))
    for tau in tau_obs:
        outputName = rootName+'Tau_'+str(tau).replace('.','_')
        for i,metric in enumerate(bundleDict[outputName].metricValues):
            if type(metric) == type({}):
                metric_data[i,0] = metric['priority']
                metric_data[i,1] = metric['nObservations']

    return metric_data

def evalFootprintPriority(opsim_db, map_data, runName, mapName,
                            n_visits_thresholds, bundleDict, logfile):
    """The bundleDict returned by the MAF structures the data as a dictionary,
    where each of the values is a MAF object.  The data from the metric is accessed
    through the metricValues method of this object.  The following shows the
    data structure of metricValues in each case.

    <runName>_GalplaneFootprintMetric_<mapName>_fiveSigmaDepth_gt_21_5_HEAL:
        list of dictionaries containing the properties
        {'nObservations': xxx, 'nObsPriority': xxx, 'map_priority': xxx}
        for each HEALpixel,
    GalplaneFootprintMetric_<mapName>_NObs:
        map of the nObservations over all HEALpixels
    GalplaneFootprintMetric_<mapName>_NObsPriority:
        map of the product of (nObservations*priority) per HEALpix over all HEALpixels
    GalplaneFootprintMetric_<mapName>_Tau_<tau_obs_value>:
        map of the HEALpix values above threshold of nObservations for each value of tau_obs
    """

    # Rootname of the metric for this map:
    rootName = 'GalplaneFootprintMetric_'+mapName+'_'

    # Calculating the ideal total priority over all pixels, if each
    # pixel were to receive at least the minimum number of visits:
    XXX This needs to take LSST visibility into account
    XXX Calculate number of HEALpix in map
    ideal_footprint_priority = map_data.sum()

    for i in range(0,len(tau_obs),1):
        ### NObs per HEALpix as a function of tau_obs
        outputName = rootName+'Tau_'+str(tau_obs[i]).replace('.','_')
        metricData = bundleDict[outputName].metricValues

        # Using the HEALpix map of the survey footprint which received
        # above the threshold number of observations for this value of tau,
        # sum the combined pixel priority from the observed pixels:
        observed_pixels = np.where(metricData > 0.0)[0]
        footprint_priority = metricData.sum()
        nObservedPixels = len(observed_pixels)

        # Figure of Merit 1 evaluates the percentage of pixels in the desired
        # survey footprint with sufficient observations
        # Figure of Merit 2 evaluates the percentage of the pixel priority
        # that received sufficient observations
        FoM1 = (nObservedPixels / nPixelFootprint) * 100.0
        FoM2 = (footprint_priority / ideal_footprint_priority) * 100.0

        # Plot sky map comparing the summed priority per pixel with the ideal
        #diff_map = -map_data
        #diff_map[observed_pixels] = metric_data[observed_pixels,0]
        file_name = runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs[i])+'_obs_footprint_priority.png'
        plot_map_data(metricData, file_name)

        ### NObsPriority per HEALpix as a function of tau_obs
        outputName = rootName+'NObsPriority'
        metricData = bundleDict[outputName].metricValues

        # Ideal value of NObsPriority per HEALpix for the current tau_obs
        XXX Take LSST visibility into account here
        expected_nvisits = nvisits_tau_obs[tau_obs[i]]
        ideal_metric_data = expected_nvisits * map_data

        # Calculate the difference per HEALpix between the metric values and
        # the ideal values
        percent_metric_data = ideal_metric_data/metricData
        file_name = runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs[i])+'_%nobspriority.png'
        plot_map_data(percent_metric_data, file_name, range=[0.0,100.0])
        file_name = runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs[i])+'_nobspriority.png'
        plot_map_data(metricData, file_name)
        file_name = runName.replace('.','_')+'_'+mapName+'_'+str(tau_obs[i])+'_ideal_nobspriority.png'
        plot_map_data(ideal_metric_data, file_name)

        # Figure of Merit 3 gives the achieved (nValidObs * HEALpix priority) as
        # a function of the ideal survey value
        FoM3 = (metricData.sum()/ideal_metric_data.sum())*100.0

        logfile.write(runName+' '+mapName+' '+str(tau_obs[i])+' '+\
                        str(round(n_visits_thresholds[i],0))+' '+str(len(observed_pixels))+' '+str(round(footprint_priority,0))+' '+\
                        str(round(ideal_footprint_priority,0))+' '+str(round(FoM,1))+' '+str(round(FoM2,1))+'\n')

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

def calcMetric(opsim_db, runName, mapName, diagnostics=False):

    bundleList = []
    metric1 = galacticPlaneMetrics.GalPlaneFootprintMetric(science_map=mapName)
    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    plotDict = {'colorMax': 950}
    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
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
    eval_footprint_priority()
