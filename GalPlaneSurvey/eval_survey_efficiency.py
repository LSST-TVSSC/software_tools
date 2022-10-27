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
import compare_survey_footprints

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

class FiguresOfMerit():
    def __init__(self):
        self.Nvisits_median = None
        self.Nvisits_stddev = None
        self.shutterFrac_median = None
        self.shutterFrac_stddev = None

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = compare_survey_footprints.load_map_data(params['map_file_path'])
    #mapName = os.path.basename(params['map_file_path'].replace('.fits',''))
    print('Total number of pixels in map: '+str(len(map_data_table)))

    # Assign threshold numbers of visits:
    n_visits_thresholds = compare_survey_footprints.calcNVisitsThresholds()

    # Compute metrics:
    bundleDict = calcMetrics(params, opsim_db, runName, diagnostics=True)

    # Start logging, and loop metric over all science maps:
    logfile = open(os.path.join(params['output_dir'],runName+'_survey_efficiency_data.txt'),'w')
    logfile.write('# runName   mapName  nVisits_median  nVisits_stddev  shutterFrac_median shutterFrac_stddev\n')

    for column in map_data_table.columns:
        mapName = column.name

        # Extract the map data for the current science map:
        map_data = getattr(map_data_table, mapName)
        print('Calculating survey region overlap for '+mapName)

        # Fetch the HEALpix indices of the desired science survey region:
        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data)

        # Calculate Figures of Merit:
        FoM = FiguresOfMerit()
        FoM = eval_shutter_fraction(runName, desired_healpix, bundleDict, FoM)
        FoM = eval_nvisits(runName, bundleDict, desired_healpix, FoM)

        # Record to the log:
        logfile.write(runName+' '+mapName+' '+\
                        str(FoM.Nvisits_median)+' '+\
                        str(FoM.Nvisits_stddev)+' '+\
                        repr(FoM.shutterFrac_median)+' '+\
                        repr(FoM.shutterFrac_stddev)+'\n')

    logfile.close()

def eval_nvisits(runName, bundleDict, desired_healpix, FoM):

    outputName = runName.replace('.','_')+'_Nvis_fiveSigmaDepth_gt_21_5_HEAL'
    metricData = bundleDict[outputName].metricValues.filled(0.0)

    # Calculate the median number of visits per HEALpixel in the
    # desired survey region:
    FoM.Nvisits_median = np.median(metricData[desired_healpix])
    FoM.Nvisits_stddev = (metricData[desired_healpix]).std()
    print('-> Nvisits over survey region: median='+\
            str(FoM.Nvisits_median)+' stddev='+str(FoM.Nvisits_stddev))

    return FoM

def eval_shutter_fraction(runName, desired_healpix, bundleDict, FoM):

    outputName = runName.replace('.','_')+'_OpenShutterFraction_fiveSigmaDepth_gt_21_5_HEAL'
    metricData = bundleDict[outputName].metricValues.filled(0.0)

    # Calculate the median shutterFrac for the given desired region:
    FoM.shutterFrac_median = np.median(metricData[desired_healpix])
    FoM.shutterFrac_stddev = (metricData[desired_healpix]).std()
    print('-> Shutter open fraction over survey region: median='+\
            str(FoM.shutterFrac_median)+' stddev='+str(FoM.shutterFrac_stddev))

    return FoM

def calcMetrics(params, opsim_db, runName, diagnostics=False):

    bundleList = []
    metric1 = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
    metric2 = maf.metrics.technicalMetrics.OpenShutterFractionMetric()
    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    plotDict = {'colorMax': 950}
    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric2, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir=params['output_dir'], resultsDb=None)
    bundleGroup.runAll()

    if diagnostics:
        bundleGroup.plotAll(closefigs=False)

    return bundleDict


if __name__ == '__main__':
    params = compare_survey_footprints.get_args()
    run_metrics(params)
