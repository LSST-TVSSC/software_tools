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
from rubin_sim.maf.metrics import galplane_time_sampling_metrics
import compare_survey_footprints
import yearlyVIMMetric

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
tau_obs = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])
plot_png = False

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    #opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = load_map_data(params['map_file_path'])

    # Log file:
    logfile = open(os.path.join(params['output_dir'],runName+'_annual_cadence_data.txt'),'w')
    logfile.write('# Col 1: runName\n')
    logfile.write('# Col 2: mapName\n')
    logfile.write('# Col 3: tau\n')
    logfile.write('# Col 4: tau_var\n')
    icol = 5
    for iyear in range(1,11,1):
        logfile.write('# Col '+str(icol)+': VIM_year'+str(iyear)+'\n')
        icol += 1

    debug = open(os.path.join(params['output_dir'],'annualVIM_debug.log'),'w')
    debug.write('OpSim runName = '+runName+'\n')

    # Loop over all science survey regions:
    for column in map_data_table.columns:
        mapName = column.name
        map_data = getattr(map_data_table, mapName)
        debug.write('Map name = '+mapName)

        # Compute the metrics for the current map
        bundleDict = calc_cadence_metrics(opsim_db, runName, mapName)

        # Calculate the Rubin visibility zone:
        #rubin_visibility_zone = compare_survey_footprints.calc_rubin_visibility(bundleDict, runName)
        #debug.write('Npix Rubin visibility zone: '+str(len(rubin_visibility_zone))+'\n')
        #debug.write('Rubin visibility zone data: '+repr(rubin_visibility_zone[0:10])+'\n')

        # Determine the HEALpix index of the desired science survey region,
        # taking the Rubin visbility zone into account:
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)
        debug.write('Npix Desired healpix: '+str(len(desired_healpix))+'\n')
        debug.write('Desired healpix data: '+repr(desired_healpix[0:10])+'\n')

        # Loop over each cadence category:
        for i in range(0,len(tau_obs),1):
            tau_var = tau_obs[i]*5.0

            FoM = eval_metrics_by_region(params,bundleDict,map_data,runName,mapName,tau_obs[i],
                                         desired_healpix)
            FoM.record(logfile)

        debug.write('\n=================================\n')

    logfile.close()
    debug.close()

class VIMFigureOfMerit:
    def __init__(self):
        self.tau = None
        self.tau_var = None
        self.runName = None
        self.mapName = None

    def record(self, logfile):
        output = self.runName+' '+self.mapName+' '+str(self.tau)+' '+str(self.tau_var)
        for iyear in range(1,11,1):
            output = output + ' '+ str(getattr(self,'VIM_'+str(iyear)))

        logfile.write(output+'\n')

def eval_metrics_by_region(params,bundleDict,map_data,runName,mapName,tau,
                            desired_healpix,datalog=None):
    """Plot spatial maps of the values of both metrics, modulo each of the
    desired spatial regions, and for all four timescale categories.
    Sum the metrics over the desired survey region, and compare with ideal value
    of 1*survey region NHealpix.
    """
    # Intialize plot figure:
    plt.rcParams['text.color'] = "#1f1f1f"
    plt.rcParams['font.size'] = 22

    FoM = VIMFigureOfMerit()
    FoM.tau = tau
    FoM.tau_var = tau*5.0
    FoM.runName = runName
    FoM.mapName = mapName

    outputName1 = runName.replace('.','_')+'_YearlyVIMMetric_'+mapName+'_fiveSigmaDepth_gt_21_5_HEAL'
    metricData = bundleDict[outputName1].metric_values.filled(0.0)

    sums = np.zeros(10)

    # Sum the value of the metric over all pixels in the current map,
    # for each year of the survey
    for i in range(0,NPIX,1):
        if type(metricData[i]) == type({}):
            for iyear in range(1,11,1):
                if iyear in metricData[i].keys():
                    sums[(iyear-1)] += metricData[i][iyear][tau]

    # Metric value per year is averaged over the number of pixels in the map
    for iyear in range(0,10,1):
        setattr(FoM, 'VIM_'+str(iyear+1), sums[iyear]/len(desired_healpix))

    return FoM

def calc_VIM(params,runName,mapName,tau,bundleDict,desired_healpix,FoM):
    # Intialize plot figure:
    plt.rcParams['text.color'] = "#1f1f1f"
    plt.rcParams['font.size'] = 22

    outputName1 = 'YearlyVIMMetric_'+mapName+'_Tau_'+str(tau).replace('.','_')
    metricData = bundleDict[outputName1].metric_values.filled(0.0)

    min = metricData[desired_healpix].min()
    max = metricData[desired_healpix].max()
    if plot_png:
        file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_'+str(tau)+'_yearlyVIM.png')
        compare_survey_footprints.plot_map_data(metricData, file_name, range=[min,max])

    FoM.sumVIM = metricData[desired_healpix].sum()
    FoM.medianVIM = np.median(metricData[desired_healpix])
    diff = (metricData[desired_healpix] - FoM.medianVIM)
    FoM.stddevVIM = np.sqrt( (diff*diff).sum()/float(len(desired_healpix)) )
    ideal_sum = float(len(desired_healpix))
    FoM.percent_sumVIM = (FoM.sumVIM / ideal_sum)*100.0

    Fn14 = galplane_time_sampling_metrics.calc_interval_decay(np.array([tau*4.0]), tau)[0]
    Fn12 = galplane_time_sampling_metrics.calc_interval_decay(np.array([tau*2.0]), tau)[0]
    Fn34 = galplane_time_sampling_metrics.calc_interval_decay(np.array([tau*4.0/3.0]), tau)[0]
    FoM.VIMdecayFn14 = Fn14*float(len(desired_healpix))
    FoM.VIMdecayFn12 = Fn12*float(len(desired_healpix))
    FoM.VIMdecayFn34 = Fn34*float(len(desired_healpix))

    return FoM

def plot_cumulative_VIM(params,runName,mapName,map_data,bundleDict,desired_healpix):
    tau_colours3 = {2.0: '#26C6DA', 5.0: '#112E51', 11.0: '#FF7043',
                    20.0: '#78909C', 46.5: '#2E78D2', 73.0: '#FFBEA9'}

    if 'combined' in mapName:
        fig = plt.figure(1,figsize=(10,10))
        plt.rcParams['font.size'] = 22
        ax = plt.subplot(111)
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

        xdata = map_data
        idx = np.argsort(xdata)
        cumData = {'x': xdata[idx]}
        for tau in tau_obs:
            outputName1 = 'GalPlaneVisitIntervalsTimescales_'+mapName+'_Tau_'+str(tau).replace('.','_')
            metricData = bundleDict[outputName1].metric_values.filled(0.0)

            ydata = np.cumsum(metricData[idx])

            plt.plot(xdata[idx], ydata, color=tau_colours3[tau], ls='-', label='$\\tau_{obs}$='+str(tau))

            cumData[tau] = ydata

        idealData = np.zeros(len(metricData))
        idealData[desired_healpix] = 1.0
        ydata = np.cumsum(idealData[idx])
        plt.plot(xdata[idx], ydata, color='k', ls='-.', label='Ideal')
        cumData['ideal'] = ydata

        plt.xlabel('HEALpixel priority')
        plt.ylabel('Cumulative VIM')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(params['output_dir'], runName+'_'+mapName+'_cumulativeVIM.png'))
        plt.close(1)

        datafile = open(os.path.join(params['output_dir'], runName+'_'+mapName+'_cumulativeVIM.dat'),'w')
        for i in idx:
            entry = str(xdata[i])+' '+' '.join(str(cumData[tau][i]) for tau in tau_obs)+'\n'
            datafile.write(entry)
        datafile.close()

def calc_SVGM(params,runName,mapName,tau,bundleDict,desired_healpix,FoM,datalog=None):
    # Intialize plot figure:
    plt.rcParams['text.color'] = "#1f1f1f"
    plt.rcParams['font.size'] = 22

    # Season Gaps metric uses tau_obs*5 as input timescales because this
    # metric is only relevant to longer timescale variables.
    tau_var = tau * 5.0

    outputName2 = 'GalPlaneSeasonGapsTimescales_'+mapName+'_Tau_'+str(tau_var).replace('.','_')
    metricData = bundleDict[outputName2].metric_values.filled(0.0)
    if datalog:
        datalog.write('calc_SVGM: metricData '+repr(metricData[0:10])+'\n')
        datalog.write('calc_SVGM: metricData range='+str(metricData.min())+' '+str(metricData.max())+'\n')
        datalog.write('calc_SVGM: desired_pixels '+repr(desired_healpix[0:10])+'\n')

    #metric2_healpix = np.zeros(NPIX)
    #metric2_healpix[desired_healpix] = metricData[desired_healpix]

    # Filter out extreme values.  This metric should return values between 0 to <<100.
    # However, in practise extreme values can occur, normally at the edge of the
    # Rubin visibility zone.
    #idx = np.argwhere(metric2_healpix < 0)
    #if datalog:
    #    datalog.write('calc_SVGM: metric2 out-of-bounds values: '+repr(idx)+'\n')
    #metric2_healpix[idx] = 0.0
    #idx = np.argwhere(metric2_healpix > 100)
    #metric2_healpix[idx] = 0.0

    if datalog:
        datalog.write('calc_SVGM: metric2 out-of-bounds values: '+repr(idx)+'\n')
        datalog.write('calc_SVGM: metric2 '+repr(metricData[desired_healpix][0:10])+'\n')
        datalog.write('calc_SVGM: metric2 range='+str(metricData[desired_healpix].min())+' '+str(metricData[desired_healpix].max())+'\n')
        idx = np.argwhere(np.isnan(metricData[desired_healpix]))
        datalog.write('calc_SVGM: NaN values metric 2: '+repr(idx)+'\n')


    min = metricData[desired_healpix].min()
    max = metricData[desired_healpix].max()
    if plot_png:
        file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_'+str(tau_var)+'_calcSVGM.png')
        compare_survey_footprints.plot_map_data(metricData, file_name, range=[min,max])

    FoM.sumSVGM = metricData[desired_healpix].sum()
    ideal_sum = float(len(desired_healpix))
    FoM.percent_sumSVGM = (FoM.sumSVGM / ideal_sum)*100.0
    if datalog:
        datalog.write('calc_SVGM: sumSVGM '+str(FoM.sumSVGM)+'\n')
        datalog.write('calc_SVGM: ideal_sum '+str(ideal_sum)+'\n')
        datalog.write('calc_SVGM: FoM.percent_sumSVGM '+str(FoM.percent_sumSVGM)+'\n')

    return FoM

def calc_VIP(params,runName,mapName,map_data,tau,bundleDict,desired_healpix,FoM):
    # Intialize plot figure:
    plt.rcParams['text.color'] = "#1f1f1f"
    plt.rcParams['font.size'] = 22

    outputName1 = 'GalPlaneVisitIntervalsTimescales_'+mapName+'_Tau_'+str(tau).replace('.','_')
    metricData = bundleDict[outputName1].metric_values.filled(0.0)

    metric1_healpix = np.zeros(NPIX)
    metric1_healpix[desired_healpix] = metricData[desired_healpix]

    metric1_priority = np.zeros(NPIX)
    metric1_priority[desired_healpix] = metricData[desired_healpix] * map_data[desired_healpix]

    min = metric1_priority[desired_healpix].min()
    max = metric1_priority[desired_healpix].max()
    if plot_png:
        file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_'+str(tau)+'_calcVIP.png')
        compare_survey_footprints.plot_map_data(metric1_priority, file_name, range=[min,max])

    FoM.sumVIP = metric1_priority[desired_healpix].sum()
    ideal_priority = map_data[desired_healpix].sum()
    FoM.percent_sumVIP = (FoM.sumVIP / ideal_priority)*100.0

    Fn14 = galplane_time_sampling_metrics.calc_interval_decay(np.array([tau*4.0]), tau)
    Fn12 = galplane_time_sampling_metrics.calc_interval_decay(np.array([tau*2.0]), tau)
    Fn34 = galplane_time_sampling_metrics.calc_interval_decay(np.array([tau*4.0/3.0]), tau)
    FoM.VIPdecayFn14 = (Fn14*map_data[desired_healpix]).sum()
    FoM.VIPdecayFn12 = (Fn12*map_data[desired_healpix]).sum()
    FoM.VIPdecayFn34 = (Fn34*map_data[desired_healpix]).sum()

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

def calc_cadence_metrics(opsim_db, runName, mapName, diagnostics=False):

    bundleList = []

    metric = yearlyVIMMetric.YearlyVIMMetric(science_map=mapName)

    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.healpix_slicer.HealpixSlicer(nside=NSIDE, use_cache=False)
    plotDict = {'colorMax': 950}

    bundleList.append(maf.metric_bundles.metric_bundle.MetricBundle(metric, slicer, constraint, run_name=runName, plot_dict=plotDict))
    bundleDict = maf.metric_bundles.metric_bundle_group.make_bundles_dict_from_list(bundleList)
    bundleGroup = maf.metric_bundles.metric_bundle_group.MetricBundleGroup(bundleDict, opsim_db, out_dir='test', results_db=None)
    bundleGroup.run_all()

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
