################################################################################################
# Metric to evaluate the Galactic Science Figure of Merit
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
from os import path
from sys import argv
import healpy as hp
import rubin_sim.maf as maf
from rubin_sim.maf.metrics import galacticPlaneMetrics, galplaneTimeSamplingMetrics
import compare_survey_footprints, eval_survey_cadence, eval_time_per_filter

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
output_dir = './FoM_results'

class FigureOfMerit:
    def __init__(self):
        self.tau_obs = None
        self.tau_var = None
        self.runName = None
        self.mapName = None
        self.filters = ["u", "g", "r", "i", "z", "y"]
        for f in self.filters:
            setattr(self,f,{'mean_fexpt_ratio': None,
                            'stddev_fexpt_ratio': None,
                            'pix_obs': None,
                            'npix_percent': None,
                            'galSciFoM': None})
        self.region_priority_percent = None
        self.footprint_priority = None
        self.ideal_footprint_priority = None
        self.sumVIM = None
        self.percent_sumVIM = None
        self.galSciFoM = None

    def recordToLog(self,logfile):
        for f in self.filters:
            data = getattr(self,f)
            logfile.write(self.runName+' '+self.mapName+' '+str(self.tau_obs)+' '+f+' '+\
                        repr(self.footprint_priority)+' '+\
                        repr(self.ideal_footprint_priority)+' '+\
                        repr(self.region_priority_percent)+' '+\
                        str(self.sumVIM)+' '+\
                        str(self.percent_sumVIM)+' '+\
                        str(round(data['mean_fexpt_ratio'],1))+' '+\
                        str(round(data['stddev_fexpt_ratio'],1))+' '+\
                        str(data['pix_obs'])+' '+\
                        str(round(data['npix_percent'],1))+' '+\
                        str(data['galSciFoM'])+'\n')

    def calcFoM(self):
        for f in self.filters:
            data = getattr(self,f)
            data['galSciFoM'] = 0.0
            data['galSciFoM'] += self.region_priority_percent/100.0 * \
                            self.percent_sumVIM/100.0 * \
                                data['npix_percent']/100.0
            setattr(self,f,data)

def calcGalScienceFoM():

    params = get_args()

    # Load the current OpSim database
    runName = path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = compare_survey_footprints.load_map_data(params['map_file_path'])
    #mapName = path.basename(params['map_file_path'].replace('.fits',''))
    print('Total number of pixels in map: '+str(len(map_data_table)))

    # Start logging, and loop metric over all science maps:
    logfile = open(path.join(output_dir,runName+'_galScienceFoM_data.txt'),'w')
    logfile.write('# runName   mapName  tau_obs  filter footprint_priority  ideal_footprint_priority  %ofFootprintPriority  Sum(VisitIntervalMetric) %OfIdeal(VisitIntervalMetric) mean(fexpt_ratio)  stddev(fexpt_ratio)  npix_obs  %OfIdeal(FilterRatio)  GalScienceFoM\n')
    for column in map_data_table.columns:
        mapName = column.name

        # Extract the map data for the current science map:
        map_data = getattr(map_data_table, mapName)
        print('Calculating survey region overlap for '+mapName)

        # Calculate the metrics for this science map
        bundleDict = calcMetrics(opsim_db, runName, mapName)
        print(bundleDict.keys())

        # Determine the HEALpix index of the desired science survey region,
        # taking the Rubin visbility zone into account:
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

        # Loop over each cadence category:
        for i in range(0,len(tau_obs),1):

            # Instantiate FiguresOfMerit object to hold results of analysis:
            FoM = FigureOfMerit()
            FoM.runName = runName
            FoM.mapName = mapName
            FoM.tau_obs = tau_obs[i]
            FoM.tau_var = FoM.tau_obs*5.0

            # Sum the HEALpixel priorities of all adequately sampled pixels
            # from the survey region
            FoM = compare_survey_footprints.eval_footprint_priority(map_data, runName, mapName,
                                          tau_obs[i], desired_healpix,
                                          bundleDict, FoM)

            # Calculate the Visit Interval Metric for all pixels in the
            # desired region
            FoM = eval_survey_cadence.calc_VIM(runName,mapName,tau_obs[i],bundleDict,desired_healpix,FoM)

            # Calculate the percentage of pixels to recieve the desired
            # filter cadence:
            FoM =  eval_time_per_filter.eval_filters_by_region(params,bundleDict,runName,
                                        mapName,map_data,desired_healpix,FoM)

            # Calculate combined Figure of Merit
            FoM.calcFoM()

            # Record results:
            FoM.recordToLog(logfile)

    logfile.close()

def calcMetrics(opsim_db, runName, mapName):

    bundleList = []

    metric1 = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
    metric2 = galacticPlaneMetrics.GalPlaneFootprintMetric(science_map=mapName)
    metric3 = galplaneTimeSamplingMetrics.GalPlaneVisitIntervalsTimescaleMetric(science_map=mapName)
    metric4 = galplaneTimeSamplingMetrics.GalPlaneSeasonGapsTimescaleMetric(science_map=mapName)
    metric5 = galacticPlaneMetrics.GalPlaneTimePerFilterMetric(science_map=mapName)

    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    plotDict = {'colorMax': 950}

    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric2, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric3, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric4, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric5, slicer, constraint, runName=runName, plotDict=plotDict))

    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    return bundleDict

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['map_file_path'] = input('Please enter the path to the GP footprint map: ')
    else:
        params['opSim_db_file'] = argv[1]
        params['map_file_path'] = argv[2]

    return params

if __name__ == '__main__':
    calcGalScienceFoM()
