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
import compare_survey_footprints, eval_survey_cadence
import test_compare_survey_footprint
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

@pytest.mark.skip(reason="Experimental code")
def test_metric_calculation():

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(TEST_DB_PATH)

    map_data_table = compare_survey_footprints.load_map_data(TEST_MAP_PATH)
    #print(map_data_table.columns)
    mapName = 'combined_map'
    tau_obs = 11.0
    tau_var = tau_obs*5.0

    log = open('./cadence_results/test_data.log','w')

    nLoop = 10
    for i in range(0,nLoop,1):
        log.write('mapName: '+mapName+'\n')
        log.write('runName: '+runName+'\n')
        log.write('tau_obs: '+str(tau_obs)+'\n')

        FoM = eval_survey_cadence.TimeFiguresOfMerit()

        bundleDict = eval_survey_cadence.calc_cadence_metrics(opsim_db, runName, mapName)

        outputName1 = 'GalPlaneVisitIntervalsTimescales_'+mapName+'_Tau_'+str(tau_obs).replace('.','_')
        outputName2 = 'GalPlaneSeasonGapsTimescales_'+mapName+'_Tau_'+str(tau_var).replace('.','_')

        metric1_data = bundleDict[outputName1].metricValues
        metric2_data = bundleDict[outputName2].metricValues

        np.savetxt(os.path.join('./cadence_results/', outputName1+'_run'+str(i)+'.txt'), metric1_data)
        np.savetxt(os.path.join('./cadence_results/', outputName2+'_run'+str(i)+'.txt'), metric2_data)

        log.write('OutputName: '+outputName1+'\n')
        log.write('OutputName: '+outputName2+'\n')

        log.write('metricData 1: '+repr(metric1_data)+'\n')
        log.write('metricData 2: '+repr(metric2_data)+'\n')

        rubin_visibility_zone = compare_survey_footprints.calc_rubin_visibility(bundleDict, runName)
        log.write('rubin_visibility_zone: '+repr(rubin_visibility_zone)+'\n')
        log.write('Npix rubin_visibility_zone: '+str(len(rubin_visibility_zone))+'\n')

        map_data = getattr(map_data_table, mapName)
        log.write('map_data: '+repr(map_data)+'\n')
        log.write('Npix map_data: '+str(len(map_data))+'\n')

        desired_healpix = compare_survey_footprints.calc_desired_survey_map(mapName, map_data, rubin_visibility_zone)

        log.write('N survey pixels: '+str(len(desired_healpix))+'\n')
        log.write('Desired survey pixels: '+repr(desired_healpix[0:10])+'\n')

        FoM = eval_survey_cadence.eval_metrics_by_region(bundleDict,map_data,runName,mapName,tau_obs,
                                rubin_visibility_zone, desired_healpix, datalog=log)

        log.write('VIM: '+str(FoM.sumVIM)+'\n')
        log.write('Percent VIM: '+str(FoM.percent_sumVIM)+'\n')
        log.write('SVGM: '+str(FoM.sumSVGM)+'\n')
        log.write('Percent SVGM: '+str(FoM.percent_sumSVGM)+'\n')
        log.write('VIP: '+str(FoM.sumVIP)+'\n')
        log.write('Percent VIP: '+str(FoM.percent_sumVIP)+'\n')

        idx = np.argwhere(np.isnan(metric1_data))
        log.write('NaN values metric 1: '+repr(idx)+'\n')
        log.write('metricData entries: '+repr(metric1_data[idx])+'\n')
        log.write('Metric sum values: '+repr(metric1_data.sum())+'\n')
        log.write('-------------------------------------------\n')

    log.close()

def test_calc_VIM():

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    mapName = 'combined'
    tau_obs = 11.0
    outputName = 'GalPlaneVisitIntervalsTimescales_'+mapName+'_Tau_'+str(tau_obs).replace('.','_')
    bundleDict = test_compare_survey_footprint.simulate_bundleDict(outputName)

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

    FoM = eval_survey_cadence.TimeFiguresOfMerit()
    FoM.tau = tau_obs
    FoM.tau_var = tau_obs*5.0

    FoM = eval_survey_cadence.calc_VIM(runName,mapName,tau_obs,bundleDict,desired_healpix,FoM)

    assert(FoM.sumVIM == nMetricValues)
    assert(FoM.percent_sumVIM == (float(nMetricValues)/float(nSurveyRegion))*100.0)


def test_calc_SVGM():

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    mapName = 'combined'
    tau_obs = 11.0
    tau_var = tau_obs * 5.0
    outputName = 'GalPlaneSeasonGapsTimescales_'+mapName+'_Tau_'+str(tau_var).replace('.','_')
    bundleDict = test_compare_survey_footprint.simulate_bundleDict(outputName)

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

    FoM = eval_survey_cadence.TimeFiguresOfMerit()
    FoM.tau = tau_obs
    FoM.tau_var = tau_var

    FoM = eval_survey_cadence.calc_SVGM(runName,mapName,tau_obs,bundleDict,desired_healpix,FoM)

    assert(FoM.sumSVGM == nMetricValues)
    assert(FoM.percent_sumSVGM == (float(nMetricValues)/float(nSurveyRegion))*100.0)

def test_calc_VIP():

    runName = os.path.split(TEST_DB_PATH)[-1].replace('.db', '')
    mapName = 'combined'
    tau_obs = 11.0
    outputName = 'GalPlaneVisitIntervalsTimescales_'+mapName+'_Tau_'+str(tau_obs).replace('.','_')
    bundleDict = test_compare_survey_footprint.simulate_bundleDict(outputName)

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

    map_data = np.zeros(NPIX)
    map_data[desired_healpix] = 1.0

    FoM = eval_survey_cadence.TimeFiguresOfMerit()
    FoM.tau = tau_obs
    FoM.tau_var = tau_obs*5.0

    FoM = eval_survey_cadence.calc_VIP(runName,mapName,map_data,
                                        tau_obs,bundleDict,desired_healpix,FoM)

    assert(FoM.sumVIP == nMetricValues)
    assert(FoM.percent_sumVIP == (float(nMetricValues)/float(nSurveyRegion))*100.0)


if __name__ == '__main__':
    test_metric_calculation()
    #test_calc_VIM()
    #test_calc_SVGM()
    #test_calc_VIP()
