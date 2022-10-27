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


config = {'star_map_dir': '/Users/rstreet1/rubin_sim_data/maps/TriMaps',
        'GP': {'data_file': 'TRIstarDensity_r_nside_64_ext.npz'}}
star_density_map = generate_sky_maps.load_star_density_data(config,limiting_mag=27.5)
hp_star_density = generate_sky_maps.rotateHealpix(star_density_map)
idx = hp_star_density > 0.0
hp_log_star_density = np.zeros(len(hp_star_density))
hp_log_star_density[idx] = np.log10(hp_star_density[idx])


fig = plt.figure(3,(10,10))
hp.mollview(hp_log_star_density, title='Density of stars across the sky',
      cmap=cm.Greys)
hp.graticule()
plt.tight_layout()
file_path = './footprint_maps/stellar_density_all_sky.png'
plt.savefig(file_path)
plt.close(3)
