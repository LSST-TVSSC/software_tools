import os
import copy
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
output_dir = 'galplane_test_results'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def calc_cadence_metrics(opsim_db, runName, mapName, diagnostics=False):
    bundleList = []
    metric1 = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
    metric2 = maf.GalPlaneVisitIntervalsTimescaleMetric(science_map=mapName)
    metric3 = maf.GalPlaneSeasonGapsTimescaleMetric(science_map=mapName)
    constraint = 'fiveSigmaDepth > 21.5'
    slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
    plotDict = {'colorMax': 950}
    bundleList.append(maf.MetricBundle(metric1, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric2, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleList.append(maf.MetricBundle(metric3, slicer, constraint, runName=runName, plotDict=plotDict))
    bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
    bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir=output_dir, resultsDb=None, saveEarly=False)
    bundleGroup.runAll()
    if diagnostics:
        bundleGroup.plotAll(closefigs=False)
    return bundleDict

TEST_DB_PATH = get_baseline()
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
    metric1_data = bundleDict[outputName1].metricValues.filled(0)
    metric2_data = bundleDict[outputName2].metricValues.filled(0)
    np.savetxt(os.path.join(output_dir, outputName1+'_run'+str(i)+'.txt'), metric1_data)
    np.savetxt(os.path.join(output_dir, outputName2+'_run'+str(i)+'.txt'), metric2_data)
    if i > 0:
        assert(np.array_equal(metric1_data,old_metric1_data))
        assert(np.array_equal(metric2_data,old_metric2_data))
    old_metric1_data = copy.deepcopy(metric1_data)
    old_metric2_data = copy.deepcopy(metric2_data)
