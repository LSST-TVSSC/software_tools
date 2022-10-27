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
from sys import path as pythonpath
from rubin_sim.maf.metrics import galacticPlaneMetrics
from rubin_sim.maf.metrics import galplaneTimeSamplingMetrics
import pytest
import copy

TEST_DB_PATH = '/Users/rstreet1/rubin_sim_data/sim_baseline/baseline_v2.0_10yrs.db'
#TEST_DB_PATH = '/Users/rstreet1/rubin_sim_data/sims/baseline_v1.5_10yrs.db'
TEST_MAP_PATH = '/Users/rstreet1/rubin_sim_data/maf/priority_GalPlane_footprint_map_data_sum.fits'
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
output_dir = './cadence_results/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def calc_cadence_metrics(opsim_db, runName, mapName, diagnostics=False):

    bundleList = []

    metric1 = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
    metric2 = galplaneTimeSamplingMetrics.GalPlaneVisitIntervalsTimescaleMetric(science_map=mapName)
    metric3 = galplaneTimeSamplingMetrics.GalPlaneSeasonGapsTimescaleMetric(science_map=mapName)

    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    plotDict = {'colorMax': 950}

    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric2, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric3, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir='test', resultsDb=None)
    bundleGroup.runAll()

    if diagnostics:
        bundleGroup.plotAll(closefigs=False)

    return bundleDict


runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
opsim_db = maf.OpsimDatabase(TEST_DB_PATH)
mapName = 'combined_map'
tau_obs = 11.0
tau_var = tau_obs * 5.0

old_metric1_data = None
old_metrid2_data = None

nLoop = 10
for i in range(0,nLoop,1):
    bundleDict = calc_cadence_metrics(opsim_db, runName, mapName)

    outputName1 = 'GalPlaneVisitIntervalsTimescales_'+mapName+'_Tau_'+str(tau_obs).replace('.','_')
    outputName2 = 'GalPlaneSeasonGapsTimescales_'+mapName+'_Tau_'+str(tau_var).replace('.','_')

    metric1_data = bundleDict[outputName1].metricValues
    metric2_data = bundleDict[outputName2].metricValues

    np.savetxt(os.path.join('./cadence_results/', outputName1+'_run'+str(i)+'.txt'), metric1_data)
    np.savetxt(os.path.join('./cadence_results/', outputName2+'_run'+str(i)+'.txt'), metric2_data)

    if i > 0:
        assert(np.array_equal(metric1_data,old_metric1_data))
        assert(np.array_equal(metric2_data,old_metric2_data))

    old_metric1_data = copy.deepcopy(metric1_data)
    old_metric2_data = copy.deepcopy(metric2_data)
