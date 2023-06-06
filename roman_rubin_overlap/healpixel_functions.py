import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp

def skycoord_to_HPindex(coord, NSIDE, radius=1.0):
    '''Function to calculate the HEALpixel index of a SkyCoord object
    using the ICRS reference frame.  Radius, if specified, should be in
    degrees'''

    phi = np.deg2rad(coord.ra.deg)
    theta = (np.pi/2.0) - np.deg2rad(coord.dec.deg)
    xyz = hp.ang2vec(theta, phi)
    HPindices = hp.query_disc(NSIDE, xyz, np.deg2rad(radius))

    return HPindices

def HPindex_to_skycoord(HPindex, NSIDE):

    theta, phi = hp.pix2ang(nside=NSIDE, ipix=[HPindex])
    ra = np.rad2deg(phi)
    dec = np.rad2deg( ((np.pi/2.0) - theta) )
    s = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))

    return s
