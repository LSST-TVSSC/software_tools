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
from rubin_sim.utils import hpid2RaDec, equatorialFromGalactic
import healpy as hp
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
import compare_survey_footprints

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
tau_obs = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])
plot_png = False
USE_USER_SLICER = False

seed = 42
np.random.seed(seed)

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = compare_survey_footprints.load_map_data(params['map_file_path'])

    # Log file:
    logfile = open(os.path.join(params['output_dir'],runName+'_microlensing_data.txt'),'w')
    logfile.write('# Col 1: runName\n')
    logfile.write('# Col 2: mapName\n')
    logfile.write('# Col 3: tau\n')
    logfile.write('# Col 4: numberTotalEventsDetected\n')
    logfile.write('# Col 5: numberTotalEvents\n')
    logfile.write('# Col 6: percentTotalEvents\n')
    logfile.write('# Col 7: median percent events detected per HEALpixel\n')
    logfile.write('# Col 8: stddev percent events detected per HEALpixel\n')

    # For efficiency, recode this to use the UserPointSlicer, and calculate it
    # for the combined, GP, Bulge, MCs and pencilbeams maps respectively
    SCIENCE_MAPS = ['combined_map', 'galactic_plane_map','magellenic_clouds_map',
                    'galactic_bulge_map', 'pencilbeams_map']
    nevents_per_hp = 10
    t_start=1,
    t_end=3652,
    peak_times = np.random.uniform(low=t_start, high=t_end, size=nevents_per_hp)
    impact_parameters = np.random.uniform(low=0, high=1, size=nevents_per_hp)

    tau_range = [30.0, 200.0]

    # Simulate events per pixel in the combined region of interest, because this
    # is the superset of the pixels in the other maps.  This is more efficient
    # than calculating the metric for each map separately.
    mapName = 'combined_map'
    map_data = getattr(map_data_table, mapName)
    desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

    metric_maps = {}
    for tau in tau_range:
        print('Calculating microlensing metric for tau='+str(tau))
        peak_times = np.random.uniform(low=t_start, high=t_end, size=nevents_per_hp)
        impact_parameters = np.random.uniform(low=0, high=1, size=nevents_per_hp)
        crossing_times = np.array( [tau]*nevents_per_hp )

        map = np.zeros(NPIX)
        for i in range(0,nevents_per_hp,1):
            metric_data = calc_microlensing_metrics(opsim_db, runName, desired_healpix,
                                                    peak_times[i], impact_parameters[i], crossing_times[i])
            print(metric_data)
            print(np.where(metric_data == np.nan))
            map[:] += metric_data[:]

        metric_maps[tau] = map
        file_name = os.path.join(params['output_dir'], runName+'_'+mapName+'_'+str(round(tau,0))+'_microlensingDetect.png')
        compare_survey_footprints.plot_map_data(map, file_name, range=[0,nevents_per_hp])

    #Products:
    # Map of where lensing is detected, giving percentage of events detected/HEALpixel.
    # FoM = sum_over_pixels of events detected / sum_over_pixels all events
    # FoM = median and stddev in percent events per pixel
    # Questions to answer are
    # - total number of events detected
    # - total number of events within Bulge detected
    # - would we detect events from the survey cadence if they are there to be found?
    # - do we detect events across a range of lines of sight?

    # Loop over all science survey regions to calculate the metric FoM
    # within the different survey regions:
    for mapName in SCIENCE_MAPS:
        map_data = getattr(map_data_table, mapName)
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

        for tau in tau_range:
            metric_data = metric_maps[tau]
            print(metric_data[desired_healpix])

            FoM = MicrolensingFiguresOfMerit()
            FoM.mapName = mapName
            FoM.runName = runName
            FoM.tau = tau
            FoM.nTotalEventsDetected = int(metric_data[desired_healpix].sum())
            FoM.nTotalEventsSimulated = nevents_per_hp*len(desired_healpix)
            FoM.percentTotalEvents = (metric_data[desired_healpix].sum()/float(nevents_per_hp*len(desired_healpix)))*100.0
            FoM.medPercentDetectedPixel = np.median( (metric_data[desired_healpix]/float(nevents_per_hp))*100.0 )
            FoM.stdPercentDetectedPixel = ((metric_data[desired_healpix]/float(nevents_per_hp))*100.0).std()
            FoM.record(logfile)

    logfile.close()

class MicrolensingFiguresOfMerit:
    def __init__(self):
        self.tau = None
        self.tau_var = None
        self.runName = None
        self.mapName = None
        self.nTotalEventsSimulated = None
        self.nTotalEventsDetected = None
        self.percentTotalEvents = None
        self.medPercentDetectedPixel = None
        self.stdPercentDetectedPixel = None

    def record(self,logfile):
        logfile.write(self.runName+' '+self.mapName+' '+\
                        str(self.tau)+' '+str(self.nTotalEventsDetected)+' '+\
                        str(round(self.nTotalEventsSimulated,2))+' '+\
                        str(round(self.percentTotalEvents,2))+' '+\
                        str(round(self.medPercentDetectedPixel,2))+' '+\
                        str(round(self.stdPercentDetectedPixel,2))+'\n')

def get_radec_desired_pixels(desired_pixels):

    (theta,phi) = hp.pix2ang(NSIDE,desired_pixels)
    ra_desired_pixels = np.rad2deg(phi)
    dec_desired_pixels = np.rad2deg( (np.pi/2.0) - theta )

    return ra_desired_pixels, dec_desired_pixels


def regional_microlensingSlicer(desired_healpix,tau,
    t_start=1,
    t_end=3652,
    n_events=10000,
    nside=128,
    filtername="r",
):
    """
    Generate a UserPointSlicer with a population of microlensing events, suitable
    for use with the MicrolensingMetric.

    This dataSlicer has been adapted from code by Peter Yoachim, with customizations
    to generate a catalog of events for a specific region of the sky rather than
    a uniform random distribution of pointings.

    Parameters
    ----------
    desired_healpix : array of int
        HEALpixel indices of the desired survey region
    min_crossing_time : float (1)
        The minimum crossing time for the events generated (days)
    max_crossing_time : float (10)
        The max crossing time for the events generated (days)
    t_start : float (1)
        The night to start generating peaks (days)
    t_end : float (3652)
        The night to end generating peaks (days)
    n_events : int (10000)
        Number of microlensing events to generate
    seed : float (42)
        Random number seed
    nside : int (128)
        HEALpix nside, used to pick which stellar density map to load
    filtername : str ('r')
        The filter to use for the stellar density map
    """
    # This initialization now takes place outside this function as
    # the loop is structured differently
    #np.random.seed(seed)

    # We simulate n_events for each HEALpixel in the desired survey region.
    # Fixed to a single event due to external loop
    n_events = 1
    n_entries = n_events * len(desired_healpix)

    crossing_times = np.array( [tau]*n_entries )
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_entries)
    impact_paramters = np.random.uniform(low=0, high=1, size=n_entries)

    gal_l, gal_b = hpid2RaDec(NSIDE, desired_healpix, nest=True)
    ra, dec = equatorialFromGalactic(gal_l, gal_b)

    use_method_1 = False
    if use_method_1:
        (theta,phi) = hp.pix2ang(nside,desired_healpix)
        ra_desired_pixels = np.rad2deg(phi)
        dec_desired_pixels = np.rad2deg( (np.pi/2.0) - theta )

        if n_events == 1:
            ra = ra_desired_pixels
            dec = dec_desired_pixels
        else:
            ra = np.repeat(ra_desired_pixels, n_events)
            dec = np.repeat(dec_desired_pixels, n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = maf.slicers.UserPointsSlicer(ra, dec,
                                        latLonDeg=True, badval=0)

    # Add any additional information about each object to the slicer
    slicer.slicePoints["peak_time"] = peak_times
    slicer.slicePoints["crossing_time"] = crossing_times
    slicer.slicePoints["impact_parameter"] = impact_paramters

    return slicer

def HEALpixMicrolensingSlicer(t0,u0,tE,
    n_events=10000,
    nside=128,
    filtername="r",
):
    """
    Generate a UserPointSlicer with a population of microlensing events, suitable
    for use with the MicrolensingMetric.

    This dataSlicer has been adapted from code by Peter Yoachim, with customizations
    to generate a catalog of events for a specific region of the sky rather than
    a uniform random distribution of pointings.

    Parameters
    ----------
    desired_healpix : array of int
        HEALpixel indices of the desired survey region
    min_crossing_time : float (1)
        The minimum crossing time for the events generated (days)
    max_crossing_time : float (10)
        The max crossing time for the events generated (days)
    t_start : float (1)
        The night to start generating peaks (days)
    t_end : float (3652)
        The night to end generating peaks (days)
    n_events : int (10000)
        Number of microlensing events to generate
    nside : int (128)
        HEALpix nside, used to pick which stellar density map to load
    filtername : str ('r')
        The filter to use for the stellar density map
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

def calc_microlensing_metrics(opsim_db, runName, desired_healpix,
                t0, u0, tE):
    """Based on a notebook by Peter Yoachim and metric by Natasha Abrams, Markus Hundertmark
    and TVS Microlensing group:
    https://github.com/lsst/rubin_sim_notebooks/blob/main/maf/science/Microlensing%20Metric.ipynb
    """

    bundleList = []

    constraint = 'fiveSigmaDepth > 21.5'
    plotDict = {'colorMax': 950}
    metric = maf.mafContrib.microlensingMetric.MicrolensingMetric(metricCalc="detect")
    summaryMetrics = maf.batches.lightcurveSummary()

    # Microlensing metric requires a customized dataSlicer, to which the
    # desired event parameter ranges have been appended.  The metric is run
    # in detect mode, returning a boolean indicating whether or not the
    # simulated events per pixel would be detectable.
    if USE_USER_SLICER:
        slicer = regional_microlensingSlicer(desired_healpix,
                                             tau_var,
                                             n_events=1,
                                             nside=NSIDE,
                                             filtername="r")
    else:
        slicer = HEALpixMicrolensingSlicer(t0,u0,tE,
                                            n_events=1,
                                            nside=NSIDE,
                                            filtername="r")

    bundleList.append( maf.MetricBundle(metric, slicer, None, runName=runName,
                                            summaryMetrics=summaryMetrics,
                                            info_label=f'tE {tE} days') )

    # Now we can make the metric bundle, and run the metrics:
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    # Extract the values of the metric per HEALpixel as a map:
    outputName = runName.replace('.','_')+\
                '_MicrolensingMetric_detect_tE_'+\
                    str(tE).replace('.','_')+\
                        '_days_'
    if USE_USER_SLICER:
        outputName = outputName + 'USER'
        metric_data = bundleDict[outputName].metricValues
    else:
        outputName = outputName + 'HEAL'
        metric_data = bundleDict[outputName].metricValues.filled(0.0)

    return metric_data

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
