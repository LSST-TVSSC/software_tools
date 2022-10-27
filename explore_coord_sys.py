import numpy as np
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord

NSIDE = 64
def IndexToDeclRa(index):
    theta,phi=hp.pixelfunc.pix2ang(NSIDE,index)
    print('Converted index to phi, theta = ',phi,theta,'rads')
    return np.rad2deg(theta - np.pi/2.0),np.rad2deg(phi)

def DeclRaToIndex(decl,RA):
    return hp.pixelfunc.ang2pix(NSIDE,np.radians(-decl+90.),np.radians(360.-RA))

# Test location in the Galactic Bulge in decimal degrees
test_ra = (17.0 + 57.0/60.0 + 34.0/3600.0)*15.0
test_dec = (29.0 + 13.0/60.0 + 15.0/3600.0)*-1.0
print('Input test coordinates, RA, DEC: ',test_ra, test_dec, 'deg')

# Should be ring order by default
phi = np.deg2rad(test_ra)
theta = np.deg2rad(test_dec) + np.pi/2.0
print('Phi, theta = ',phi, theta,' rads')
pixels_hp = hp.ang2pix(NSIDE,theta,phi,nest=False)
print('Pixels from Healpy: ',pixels_hp)
(return_dec, return_ra) = IndexToDeclRa(pixels_hp)
print('Recalculating RA, Dec in deg from pixel index: ',return_ra, return_dec)

#(theta, phi) = hp.pix2ang(NSIDE, ipix)
#ra = np.rad2deg(phi)
#dec = np.rad2deg(0.5 * np.pi - theta)

# Astropy-healpix solution
ahp = HEALPix(nside=NSIDE, order='ring', frame=Galactic())
coords_icrs = SkyCoord( test_ra, test_dec, frame="icrs", unit=(u.deg, u.deg))
coords_gal = coords_icrs.transform_to(Galactic())
pixels_ap = ahp.skycoord_to_healpix(coords_icrs)
print('Pixels from astropy: ',pixels_ap)
