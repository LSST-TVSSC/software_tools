import numpy as np
from astropy.io import fits
from os import path
import healpy as hp

mapfile1 = '../footprint_maps/galactic_plane_map_r.fits'
mapfile2 = '../footprint_maps/v1.0.0/galactic_plane_map_r.fits'

map1 = hp.read_map(mapfile1)
map2 = hp.read_map(mapfile2)

diff_map = map2 - map1
print(diff_map)
print(diff_map.mean())
print((diff_map == 0.0).all())
