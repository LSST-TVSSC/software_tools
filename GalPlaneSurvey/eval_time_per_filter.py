import os
from sys import argv
#from sys import path as pythonpath
#pythonpath.append('/Users/rstreet1/software/rubin_sim_gal_plane/rubin_sim/maf/metrics/')
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
#from eval_survey_cadence import load_map_data, get_priority_threshold, plot_map_data
import compare_survey_footprints
from rubin_sim.maf.metrics import galactic_plane_metrics

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
#tau_obs = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])
filters = ["u", "g", "r", "i", "z", "y"]
SCIENCE_MAPS = ['combined_map', 'galactic_plane_map','magellenic_clouds_map',
                'galactic_bulge_map', 'clementini_stellarpops_map',
                'bonito_sfr_map', 'globular_clusters_map', 'open_clusters_map',
                'zucker_sfr_map', 'pencilbeams_map', 'xrb_priority_map']
plot_png = False

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    #opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = compare_survey_footprints.load_map_data(params['map_file_path'])
    print('Total number of pixels in map: '+str(len(map_data_table)))

    # Start logfile
    logfile = open(os.path.join(params['output_dir'],runName+'_time_per_filter_metric_data.txt'),'w')
    logfile.write('runName   mapName  filter   mean(fexpt_ratio)  min(fexpt_ratio)   max(fexpt_ratio)   median(fexpt_ratio)   stddev(fexpt_ratio)  npix_obs_50  %pixels_over_50 npix_obs_75  %pixels_over_75 npix_obs_90  %pixels_over_90   npix_obs_100  %pixels_over_100  %sum_ratio\n')

    # Loop over all science maps:
    for mapName in SCIENCE_MAPS:
        map_data = getattr(map_data_table, mapName)

        # Compute metrics
        bundleDict = calc_filter_metric(opsim_db, runName, mapName)
        print(bundleDict.keys())

        # Calculate the Rubin visibility zone:
        #rubin_visibility_zone = compare_survey_footprints.calc_rubin_visibility(bundleDict, runName)

        # Determine the HEALpix index of the desired science survey region,
        # taking the Rubin visbility zone into account:
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

        FoM = FilterFiguresOfMerit()
        FoM.runName = runName
        FoM.mapName = mapName

        FoM = eval_filters_by_region(params,bundleDict,runName,
                                mapName,map_data,desired_healpix,FoM)

        FoM.recordToLog(logfile)

    logfile.close()

class FilterFiguresOfMerit():
    def __init__(self):
        self.filters = ["u", "g", "r", "i", "z", "y"]
        self.runName = None
        self.mapName = None
        for f in self.filters:
            setattr(self,f,{'mean_fexpt_ratio': None,
                            'stddev_fexpt_ratio': None,
                            'pix_obs': None,
                            'npix_percent': None})

    def recordToLog(self, logfile):
        for f in self.filters:
            data = getattr(self,f)
            logfile.write(self.runName+' '+self.mapName+' '+f+' '+\
                        str(round(data['mean_fexpt_ratio'],1))+' '+\
                        str(round(data['min_fexpt_ratio'],1))+' '+\
                        str(round(data['max_fexpt_ratio'],1))+' '+\
                        str(round(data['median_fexpt_ratio'],1))+' '+\
                        str(round(data['stddev_fexpt_ratio'],1))+' '+\
                        str(data['pix_obs_50'])+' '+\
                        str(round(data['npix_50_percent'],1))+' '+\
                        str(data['pix_obs_75'])+' '+\
                        str(round(data['npix_75_percent'],1))+' '+\
                        str(data['pix_obs_90'])+' '+\
                        str(round(data['npix_90_percent'],1))+' '+\
                        str(data['pix_obs_100'])+' '+\
                        str(round(data['npix_100_percent'],1))+' '+\
                        str(round(data['sum_ratio_percent'],1))+'\n')

def eval_filters_by_region(params,bundleDict,runName,
                            mapName,map_data,desired_healpix,FoM):

    # Metric is evaluated by calculating the percentage of pixels with
    # an fexpt ratio exceeding set threshold levels, above which we consider the
    # HEALpix to have received the desired ratio of exposure time

    # Loop over each filter:
    for f in filters:

        # Retrieve the metric data in a map array:
        outputName = 'GalplaneTimePerFilter_'+mapName+'_'+f
        metric_data = bundleDict[outputName].metric_values.filled(0.0)

        # Calculate summary metrics over all HEALpix:
        fomData = {}
        fomData['mean_fexpt_ratio'] = metric_data[desired_healpix].mean()
        fomData['min_fexpt_ratio'] = metric_data[desired_healpix].min()
        fomData['max_fexpt_ratio'] = metric_data[desired_healpix].max()
        fomData['median_fexpt_ratio'] = np.median(metric_data[desired_healpix])
        fomData['stddev_fexpt_ratio'] = metric_data[desired_healpix].std()
        fomData['pix_obs_50'] = len(np.where(metric_data[desired_healpix] >= 0.5)[0])
        fomData['npix_50_percent'] = (fomData['pix_obs_50']/len(desired_healpix))*100.0
        fomData['pix_obs_75'] = len(np.where(metric_data[desired_healpix] >= 0.75)[0])
        fomData['npix_75_percent'] = (fomData['pix_obs_75']/len(desired_healpix))*100.0
        fomData['pix_obs_90'] = len(np.where(metric_data[desired_healpix] >= 0.90)[0])
        fomData['npix_90_percent'] = (fomData['pix_obs_90']/len(desired_healpix))*100.0
        fomData['pix_obs_100'] = len(np.where(metric_data[desired_healpix] >= 1.0)[0])
        fomData['npix_100_percent'] = (fomData['pix_obs_100']/len(desired_healpix))*100.0
        fomData['sum_ratio_percent'] = (metric_data[desired_healpix].sum()/len(desired_healpix))*100.0

        setattr(FoM, f, fomData)

        if plot_png:
            file_name = os.path.join(params['output_dir'],runName+'_'+mapName+'_time_per_filter_'+f+'.png')
            compare_survey_footprints.plot_map_data(metric_data, file_name, range=[0,2.0])

    return FoM

def calc_filter_metric(opsim_db, runName, mapName, diagnostics=False):
    bundleList = []

    metric1 = maf.metrics.simple_metrics.CountMetric(col=['night'], metric_name='Nvis')
    metric2 = galactic_plane_metrics.GalPlaneTimePerFilterMetric(science_map=mapName)

    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.healpix_slicer.HealpixSlicer(nside=NSIDE)
    plotDict = {'colorMax': 950}

    bundleList.append(maf.metric_bundles.metric_bundle.MetricBundle(metric1, slicer, constraint, run_name=runName, plot_dict=plotDict))
    bundleList.append(maf.metric_bundles.metric_bundle.MetricBundle(metric2, slicer, constraint, run_name=runName, plot_dict=plotDict))

    bundleDict = maf.metric_bundles.metric_bundle_group.make_bundles_dict_from_list(bundleList)
    bundleGroup = maf.metric_bundles.metric_bundle_group.MetricBundleGroup(bundleDict, opsim_db, out_dir='test',
                                        results_db=None)
    bundleGroup.run_all()

    if diagnostics:
        bundleGroup.plot_all(close_figs=False)

    return bundleDict

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['map_file_path'] = input('Please enter the path to the footprint map summed over all filters: ')
        params['output_dir'] = input('Please enter the path to the directory for output: ')
    else:
        params['opSim_db_file'] = argv[1]
        params['map_file_path'] = argv[2]
        params['output_dir'] = argv[3]

    return params



if __name__ == '__main__':
    params = get_args()
    run_metrics(params)
