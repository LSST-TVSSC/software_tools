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
output_dir = './results'

def eval_nobs_priority_function():

    params = get_args()

    # Load the current OpSim database
    runName = os.path.split(params['opSim_db_file'])[-1].replace('.db', '')
    opsim_db = maf.OpsimDatabase(params['opSim_db_file'])

    # Load the Galactic Plane Survey footprint map data
    map_data_table = compare_survey_footprints.load_map_data(params['map_file_path'])
    mapName = 'combined_map'
    map_data = getattr(map_data_table, mapName)

    # Compute metric data:
    bundleDict = calcMetrics(opsim_db, runName, mapName)

    # Plot the number of visits per HEALpixel as a function of space and
    # scientific priority:
    plot_3d_healpix_nvis_priority(runName,bundleDict, map_data)

def get_args():

    params = {}
    if len(argv) == 1:
        params['opSim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['map_file_path'] = input('Please enter the path to the GP footprint map: ')
    else:
        params['opSim_db_file'] = argv[1]
        params['map_file_path'] = argv[2]

    return params

def calcMetrics(opsim_db, runName, mapName, diagnostics=False):

    bundleList = []
    metric1 = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
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

def plot_3d_healpix_nvis_priority(runName,bundleDict, map_data):

    nvisits_per_healpix = bundleDict[runName.replace('.','_')+\
                '_Nvis_fiveSigmaDepth_gt_21_5_HEAL'].metricValues

    healpix_index = np.arange(0,NPIX,1)

    X,Y = np.meshgrid(healpix_index, map_data)
    print(X,Y)
    Z,tmp = np.meshgrid(nvisits_per_healpix, healpix_index)
    print(Z)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colormap = 'copper'
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap=colormap, edgecolor='none')
    #ax.axes.set_xlim3d(left=(t0-2.0*tE), right=(t0+2.0*tE))
    #ax.axes.set_ylim3d(bottom=0, top=10)
    #ax.axes.set_zlim3d(bottom=Z.max(), top=Z.min())

    #ax.view_init(8,-79)
    #ax.set_axis_off()
    plt.savefig(os.path.join(output_dir,runName+'_'+mapName+'_NVis_priority_3D.png'))


if __name__ == '__main__':
    eval_nobs_priority_function()
