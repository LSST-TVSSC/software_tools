from sys import argv
from os import path
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits

NSIDE=64
PixArea = hp.nside2pixarea(NSIDE,degrees=True)
print('Using map resolution NSIDE='+str(NSIDE)+', pixscale='+str(round(PixArea,1)))

if len(argv) == 1:
    mapFile = input('Please enter the path to the HEALpixel galactic map data file: ')
    visMapFile = input('Please enter the path to the Rubin visibility zone map: ')
else:
    mapFile = argv[1]
    visMapFile = argv[2]

with fits.open(mapFile) as hdul:
    mapTable = hdul[1].data

with fits.open(visMapFile) as hdul1:
    visMapTable = hdul1[1].data

# Apply visibiltiy restrictions to the priority map data:
observableMap = mapTable.combined_map * visMapTable.visibility_map

# Pre-sort map pixels into priority order:
order = observableMap.argsort()
order = np.flip(order)
prioritizedMap = observableMap[order]

def sum_priority_over_selected_area(NSIDE, mapData, priorityThreshold):
    idx = np.where(mapData >= priorityThreshold)[0]
    pixArea = hp.nside2pixarea(NSIDE,degrees=True) * len(idx)
    sumPriority = mapData[idx].sum()
    return pixArea, sumPriority

def sum_priority_over_pixels(NSIDE, npixels, prioritizedMap):
    sumPriority = prioritizedMap[0:npixels].sum()
    return sumPriority

def calc_area_npixels(NSIDE, npixels):
    return hp.nside2pixarea(NSIDE, degrees=True) * npixels

npixels = np.arange(0,len(observableMap),1)
pixArea = np.zeros(len(npixels))
sumPriority = np.zeros(len(npixels))
for i,np in enumerate(npixels):
    pixArea[i] = calc_area_npixels(NSIDE, np)
    sumPriority[i] = sum_priority_over_pixels(NSIDE, np, prioritizedMap)

(fig, ax1) = plt.subplots(1,1,figsize=(10,10))
ax1.plot(npixels, sumPriority, 'r-')
ax2 = ax1.twiny()
ax2.plot(pixArea, sumPriority, 'r-')
ax1.set_xlabel('Number of pixels in survey region',fontsize=16)
ax1.set_ylabel('Sum of map priority for survey region', fontsize=16)
ax2.set_xlabel('Area surveyed [deg$^{2}$]',fontsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax2.tick_params(axis='x', labelsize=16)
ax1.set_title('Surveying highest priority pixels first', fontsize=18)
ax1.grid()
plt.savefig(path.join(path.dirname(mapFile),'sum_priority_with_npixels_healpix.png'))
plt.close(1)
