from sys import argv
from os import path
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits

if len(argv) == 1:
    mapFile = input('Please enter the path to the Mollweide galactic map file: ')
else:
    mapFile = argv[1]

rawImage = fits.getdata(mapFile)
idx = np.where(rawImage != -np.inf)
print('Number of valid map pixels: '+str(len(idx[0])))
priorityMap = np.zeros(rawImage.shape)
priorityMap[idx] = rawImage[idx]
mapData = priorityMap.flatten()
print('Dimensions of input image: ',priorityMap.shape)

# Estimate pixel area:
pixHeight = 180.0 / priorityMap.shape[0]
pixWidth = 360.0 / priorityMap.shape[1]
pixArea = pixHeight * pixWidth
print('One pixel has area = '+str(round(pixArea,1)))

# Pre-sort map pixels into priority order:
order = mapData.argsort()
order = np.flip(order)
prioritizedMap = mapData[order]

def sum_priority_over_pixels(npixels, prioritizedMap):
    sumPriority = prioritizedMap[0:npixels].sum()
    return sumPriority

def calc_area_npixels(pixArea, npixels):
    return pixArea * npixels

npixels = np.arange(0,40000,1)
surveyArea = np.zeros(len(npixels))
sumPriority = np.zeros(len(npixels))
for i,np in enumerate(npixels):
    surveyArea[i] = calc_area_npixels(pixArea, np)
    sumPriority[i] = sum_priority_over_pixels(np, prioritizedMap)

(fig, ax1) = plt.subplots(1,1,figsize=(10,10))
ax1.plot(npixels, sumPriority, 'r-')
ax2 = ax1.twiny()
ax2.plot(surveyArea, sumPriority, 'r-')
ax1.set_xlabel('Number of pixels in survey region',fontsize=16)
ax1.set_ylabel('Sum of map priority for survey region', fontsize=16)
ax2.set_xlabel('Area surveyed [deg$^{2}$]',fontsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax2.tick_params(axis='x', labelsize=16)
ax1.set_title('Surveying highest priority pixels first', fontsize=18)
ax1.grid()
plt.savefig(path.join(path.dirname(mapFile),'sum_priority_with_npixels_moll.png'))
plt.close(1)
