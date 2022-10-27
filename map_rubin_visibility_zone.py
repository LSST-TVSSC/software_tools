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

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)

output_dir = os.path.join(os.path.expanduser("~"),'software','LSST-TVS_software_tools','footprint_maps')

opSim_db_file = os.path.join(os.path.expanduser("~"),'rubin_sim_data','sim_baseline','baseline_v2.0_10yrs.db')
runName = os.path.split(opSim_db_file)[-1].replace('.db', '')
opsim_db = maf.OpsimDatabase(opSim_db_file)

bundleList = []
metric = maf.metrics.CountMetric(col=['night'], metricName='Nvis')
constraint = 'fiveSigmaDepth > 21.5'
slicer = maf.slicers.HealpixSlicer(nside=NSIDE, useCache=False)
plotDict = {'colorMax': 950}
bundleList.append(maf.MetricBundle(metric, slicer, constraint, runName=runName, plotDict=plotDict))
bundleDict = maf.metricBundles.makeBundlesDictFromList(bundleList)
bundleGroup = maf.MetricBundleGroup(bundleDict, opsim_db, outDir=output_dir, resultsDb=None)
bundleGroup.runAll()

outputName = runName.replace('.','_')+'_Nvis_fiveSigmaDepth_gt_21_5_HEAL'
metricData = bundleDict[outputName].metricValues.filled(0.0)

vizPixels = np.where(metricData > 0)[0]

map = np.zeros(NPIX)
map[vizPixels] = 1.0
file_name = os.path.join(output_dir, runName+'_rubin_visibility_zone.png')
fig = plt.figure(1,(10,10))
hp.mollview(map)
hp.graticule()
plt.tight_layout()
plt.savefig(file_name)
plt.close(1)

hdr = fits.Header()
hdr['NSIDE'] = NSIDE
hdr['NPIX'] = hp.nside2npix(NSIDE)
hdr['MAPTITLE'] = 'Rubin Visibility Zone'
phdu = fits.PrimaryHDU(header=hdr)
hdu_list = [phdu]
col_list = [fits.Column(name='visibility_map', array=map, format='E')]
dhdu = fits.BinTableHDU.from_columns(col_list)
hdu_list.append(dhdu)
hdul = fits.HDUList(hdu_list)
file_name = os.path.join(output_dir, runName+'_rubin_visibility_zone.fits')
hdul.writeto(file_name, overwrite=True)
