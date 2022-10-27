import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from os import path, mkdir
from sys import argv, exit
from astropy import units as u
#from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
from pylab import cm
import csv
import config_utils
import json
import generate_sky_maps

# Configuration
NSIDE = 64

file_path = './footprint_maps/xray_binaries_map_i.fits'
#file_path = './footprint_maps/priority_GalPlane_footprint_map_data_sum.fits'

pix_data = hp.read_map(file_path)
#pix_data = np.log10(pix_data)
map_title = 'X-Ray Binaries'
#map_title = 'Galactic science combined region of interest'

fig = plt.figure(3,(10,10))
plot_max = pix_data.max()
if np.isnan(plot_max):
    plot_max = 1.0
plot_max = 2.0
hp.mollview(pix_data, title=map_title,
            min=0.0, max=plot_max)
hp.graticule()
plt.tight_layout()
plt.savefig(path.join('./footprint_maps','xray_binaries_map_i.png'))
#plt.savefig(path.join('./footprint_maps','priority_maps_combined.png'))
plt.close(3)
