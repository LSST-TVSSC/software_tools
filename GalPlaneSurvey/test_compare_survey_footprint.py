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
import compare_survey_footprints
import pytest

TEST_DB_PATH = '/Users/rstreet1/rubin_sim_data/sim_baseline/baseline_v2.0_10yrs.db'
TEST_DB_PATH = '/Users/rstreet1/rubin_sim_data/sims/baseline_v1.5_10yrs.db'
TEST_MAP_PATH = '/Users/rstreet1/rubin_sim_data/maf/priority_GalPlane_footprint_map_data_sum.fits'
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
nvisits_tau_obs = {2.0: 7200,
            5.0: 2880,
            11.0: 1309,
            20.0: 720,
            46.5: 310,
            73.0: 197}

class MetricResult:
    def __init__(self):
        self.metricValues = None

def simulate_bundleDict(outputName):
    bundleDict = {}
    metric = MetricResult()
    metric.metricValues = np.zeros(NPIX)
    bundleDict[outputName] = metric

    return bundleDict

def test_calcFootprintOverlap():

    FoM = compare_survey_footprints.FiguresOfMerit()

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    mapName = 'combined'
    tau_obs = 20.0
    outputName = 'GalplaneFootprintMetric_'+mapName+'_Tau_'+str(tau_obs).replace('.','_')
    bundleDict = simulate_bundleDict(outputName)

    # The metricValues contains a list of values for each HEALpix in the sky
    # Simulate metric results for some, but not all, sky pixels
    nMetricValues = 500
    metric = bundleDict[outputName]
    metric.metricValues[0:nMetricValues] = 1.0
    bundleDict[outputName] = metric

    # Simulate the pixels indices for a desired survey region, a subset of the
    # pixels for the whole sky, which for the purposes of this test partially
    # overlaps the simulated metric output:
    nSurveyRegion = 1000
    desired_healpix = np.arange(0,nSurveyRegion,1)

    FoM = compare_survey_footprints.calcFootprintOverlap(runName, mapName, tau_obs,
                             desired_healpix, bundleDict, FoM)

    assert(FoM.overlap_healpix == nSurveyRegion - nMetricValues)
    assert(FoM.missing_healpix == nSurveyRegion - nMetricValues)
    assert(FoM.missing_percent == ((nSurveyRegion - nMetricValues)/nSurveyRegion)*100.0)
    assert(FoM.overlap_percent == ((nMetricValues)/nSurveyRegion)*100.0)

def test_eval_footprint_priority():

    FoM = compare_survey_footprints.FiguresOfMerit()

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    mapName = 'combined_map'
    tau_obs = 20.0
    outputName1 = 'GalplaneFootprintMetric_'+mapName+'_Tau_'+str(tau_obs).replace('.','_')
    bundleDict = simulate_bundleDict(outputName1)
    map_data = np.zeros(NPIX)

    # The metricValues contains a list of values for each HEALpix in the sky
    # Simulate metric results for some, but not all, sky pixels
    nMetricValues = 500
    nObs = 10.0
    nIdealObs = nvisits_tau_obs[tau_obs]
    priority = 1.0

    metric = bundleDict[outputName1]
    metric.metricValues[0:nMetricValues] = priority
    bundleDict[outputName1] = metric

    # Simulate the pixels indices for a desired survey region, a subset of the
    # pixels for the whole sky, which for the purposes of this test partially
    # overlaps the simulated metric output:
    nSurveyRegion = 1000
    desired_healpix = np.arange(0,nSurveyRegion,1)
    map_data[desired_healpix] = 1.0

    FoM = compare_survey_footprints.eval_footprint_priority(map_data, runName, mapName, tau_obs,
                                desired_healpix, bundleDict, FoM)

    assert(FoM.ideal_footprint_priority == nSurveyRegion)
    assert(FoM.footprint_priority == (nSurveyRegion - nMetricValues))
    assert(FoM.region_priority_percent == ((nSurveyRegion - nMetricValues)/nSurveyRegion)*100.0)

def test_eval_footprint_nobs_priority():

    FoM = compare_survey_footprints.FiguresOfMerit()

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    mapName = 'combined_map'
    tau_obs = 20.0
    outputName = 'GalplaneFootprintMetric_'+mapName+'_NObsPriority'
    bundleDict = simulate_bundleDict(outputName)
    map_data = np.zeros(NPIX)

    # The metricValues contains a list of values for each HEALpix in the sky
    # Simulate metric results for some, but not all, sky pixels
    nMetricValues = 500
    nObs = nvisits_tau_obs[tau_obs]
    nIdealObs = nvisits_tau_obs[tau_obs]
    priority = 1.0

    metric = bundleDict[outputName]
    metric.metricValues[0:nMetricValues] = nObs*priority
    bundleDict[outputName] = metric

    # Simulate the pixels indices for a desired survey region, a subset of the
    # pixels for the whole sky, which for the purposes of this test partially
    # overlaps the simulated metric output:
    nSurveyRegion = 1000
    desired_healpix = np.arange(0,nSurveyRegion,1)
    map_data[desired_healpix] = 1.0

    FoM = compare_survey_footprints.eval_footprint_nobs_priority(map_data, runName, mapName, tau_obs,
                                desired_healpix, bundleDict, FoM)

    assert(FoM.nobs_priority_percent == ((nSurveyRegion - nMetricValues)*nObs*priority/(nSurveyRegion*nIdealObs*priority))*100.0)

@pytest.mark.skip(reason="Experimental code")
def test_NObsPriority():

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(TEST_DB_PATH)

    map_data_table = compare_survey_footprints.load_map_data(TEST_MAP_PATH)
    #print(map_data_table.columns)
    mapName = 'combined_map'
    tau_obs = 20.0

    log = open('./results/test_data.log','w')

    nLoop = 10
    for i in range(0,nLoop,1):
        log.write('mapName: '+mapName+'\n')
        log.write('runName: '+runName+'\n')
        log.write('tau_obs: '+str(tau_obs)+'\n')

        FoM = compare_survey_footprints.FiguresOfMerit()

        bundleDict = compare_survey_footprints.calcNVisits(opsim_db, runName,
                                                            mapName)

        rootName = 'GalplaneFootprintMetric_'+mapName+'_'
        #outputName = rootName+'NObsPriority'
        outputName = rootName+'Tau_'+str(tau_obs).replace('.','_')
        log.write('OutputName: '+outputName+'\n')

        metricData = bundleDict[outputName].metricValues
        log.write('metricData: '+repr(metricData)+'\n')

        rubin_visibility_zone = compare_survey_footprints.calc_rubin_visibility(bundleDict, runName)
        log.write('rubin_visibility_zone: '+repr(rubin_visibility_zone)+'\n')

        map_data = getattr(map_data_table, mapName)
        log.write('map_data: '+repr(map_data)+'\n')

        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data, rubin_visibility_zone)

        log.write('N survey pixels: '+str(len(desired_healpix))+'\n')
        #log.write('Desired survey pixels: '+repr(desired_healpix)+'\n')

        FoM = compare_survey_footprints.calcFootprintOverlap(runName, mapName,
                                    tau_obs, desired_healpix,
                                    bundleDict, FoM)

        log.write('Overlap pixels: '+str(FoM.overlap_healpix)+'\n')
        log.write('Overlap percent: '+str(FoM.overlap_percent)+'\n')
        log.write('Missing pixels: '+str(FoM.missing_healpix)+'\n')
        log.write('Missing percent: '+str(FoM.missing_percent)+'\n')

        idx = np.argwhere(np.isnan(metricData))
        log.write('NaN values: '+repr(idx)+'\n')
        log.write('metricData entries: '+repr(metricData[idx])+'\n')
        log.write('Metric sum values: '+repr(metricData.sum())+'\n')
        log.write('-------------------------------------------\n')

    log.close()

def test_map_data_ranges():

    map_data_table = compare_survey_footprints.load_map_data(TEST_MAP_PATH)

    for column in map_data_table.columns:
        map_data = map_data = getattr(map_data_table, column.name)
        print(column.name, map_data.min(), map_data.max())

def test_rubin_visibility():

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(TEST_DB_PATH)
    mapName = 'combined_map'

    log = open('./results/test_visibility_data.log','w')
    log.write('# Run   NHealpix_vis>1   NHealpix_valid\n')
    for i in range(0,10,1):
        bundleDict = compare_survey_footprints.calcNVisits(opsim_db, runName,
                                                            mapName)
        rubin_visibility_zone1,rubin_visibility_zone2 = compare_survey_footprints.calc_rubin_visibility(bundleDict, runName)

        #map = bundleDict[runName.replace('.','_')+\
        #            '_Nvis_fiveSigmaDepth_gt_21_5_HEAL'].metricValues
        #file_name = './results/rubin_visibility_map.png'
        #compare_survey_footprints.plot_map_data(map, file_name)

        log.write(str(i)+' '+str(len(rubin_visibility_zone1))+' '+str(len(rubin_visibility_zone2))+'\n')

    log.close()

if __name__ == '__main__':
    #test_eval_footprint_priority()
    #test_eval_footprint_nobs_priority()
    test_NObsPriority()
    #test_map_data_ranges()
    #test_rubin_visibility()
