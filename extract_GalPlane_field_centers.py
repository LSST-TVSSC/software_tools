from os import getenv, path
import numpy as np
import healpy as hp
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
import science_priority_regions
from astropy.io import fits

def condense_pixel_list(high_priority_regions, regions_outside_plane):
    """Since the defined science areas overlap, this function combines the
    list of HEALpixels from the different regions to output a single,
    non-duplicated, list of HEALpix"""

    pixels = np.array([],dtype='int')
    for region, dict in high_priority_regions.items():
        pixels = np.concatenate((pixels, dict['pixel_region'].flatten()))
    for region, dict in regions_outside_plane.items():
        pixels = np.concatenate((pixels,dict['pixel_region'].flatten()))

    unique_pixels = list(np.unique(np.array(pixels)))

    return unique_pixels


NSIDE=64
footprint_maps_dir = getenv('FOOTPRINT_MAPS_DIR')
map = hp.nside2npix(NSIDE)
priority_map = hp.read_map(path.join(footprint_maps_dir,'GalPlane_priority_map_r.fits'))

ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())
(high_priority_regions, regions_outside_plane) = science_priority_regions.fetch_priority_region_data(ahp)

unique_pixels = condense_pixel_list(high_priority_regions, regions_outside_plane)

coords = ahp.healpix_to_skycoord(unique_pixels)

priority_values = priority_map[unique_pixels]

output = open(path.join(footprint_maps_dir, 'GalPlane_pointings_list.txt'),'w')
output.write('# Field center RA, Dec [deg]  Priority\n')
for i,c in enumerate(coords):
    output.write(str(c.ra.deg)+' '+str(c.dec.deg)+' '+str(priority_values[i])+'\n')
output.close()
