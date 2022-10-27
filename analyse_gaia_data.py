from os import path
from sys import argv
from astropy.io import fits
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy import units as u
from astropy_healpix import HEALPix
import healpy as hp
import matplotlib.pyplot as plt
import glob
import numpy as np

def analyse_gaia_pointing_catalogs():
    if len(argv) == 1:
        data_dir = input('Please enter the path to the output data directory: ')
    else:
        data_dir = argv[1]

    catalog_list = glob.glob(path.join(data_dir, '*.fits'))

    statistics = np.zeros((len(catalog_list),6))

    for i,catalog in enumerate(catalog_list):
        (pointing, data) = read_gaia_pointing_catalog(catalog)
        (mean_parallax, stat1, stat2, stat3) = measure_parallax_distribution(data_dir,pointing, data)
        statistics[i,0] = pointing.ra.deg
        statistics[i,1] = pointing.dec.deg
        statistics[i,2] = mean_parallax
        statistics[i,3] = stat1
        statistics[i,4] = stat2
        statistics[i,5] = stat3

    plot_statistics_on_sky(data_dir, statistics)

def read_gaia_pointing_catalog(catalog_path):
    pointing = None
    data = None
    if path.isfile(catalog_path):
        hdul = fits.open(catalog_path)
        header = hdul[0].header
        pointing = SkyCoord(header['RA']*u.degree, header['DEC']*u.degree, frame='icrs')
        data = hdul[1].data
        hdul.close()
    return pointing, data

def measure_parallax_distribution(data_dir, pointing, data):

    location = (pointing.to_string('decimal')).replace(' ','_')

    # Measure the std dev of the parallax measurements for a given pointing
    idx = np.isfinite(data['parallax'])
    mean_parallax = data['parallax'][idx].mean()
    stddev_parallax = data['parallax'][idx].mean()
    stat1 = np.log10(abs(data['parallax'][idx]).min())
    stat2 = data['parallax'][idx].max()
    stat3 = abs(data['parallax'][idx]).max() - abs(data['parallax'][idx]).min()

    fig = plt.figure(1,(10,10))
    plt.hist(data['parallax'], bins=20)
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.plot([mean_parallax,mean_parallax],[ymin,ymax], 'r-')
    plt.plot([mean_parallax-stddev_parallax,mean_parallax-stddev_parallax],[ymin,ymax], 'r-.')
    plt.plot([mean_parallax+stddev_parallax,mean_parallax+stddev_parallax],[ymin,ymax], 'r-.')
    plt.xlabel('Parallax [mas]')
    plt.ylabel('Number of stars')
    plt.savefig(path.join(data_dir,'parallax_hist_'+location+'.png'))
    plt.close(1)

    plot_para_err = False
    if plot_para_err:
        fig = plt.figure(1,(10,10))
        plt.hist(data['parallax_error'], bins=20)
        plt.xlabel('Parallax error [mas]')
        plt.ylabel('Number of stars')
        plt.savefig(path.join(data_dir,'parallax_error_hist_'+location+'.png'))
        plt.close(1)

    return mean_parallax, stat1, stat2, stat3

def plot_statistics_on_sky(data_dir, statistics):

    NSIDE=64
    ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())

    coords = SkyCoord(statistics[:,0]*u.deg, statistics[:,1]*u.deg, frame='icrs')
    pixel_index = ahp.skycoord_to_healpix(coords)

    NPIX = hp.nside2npix(NSIDE)
    pixels = np.zeros((NPIX), dtype='int')

    pixels[pixel_index] = statistics[:,5]

    fig = plt.figure(1,(10,10))
    hp.mollview(pixels, title="Gaia Parallax Distribution")
    hp.graticule()
    plt.tight_layout()
    plt.savefig(path.join(data_dir,'gaia_parallax_stddev_map.png'))
    plt.close(1)

if __name__ == '__main__':
    analyse_gaia_pointing_catalogs()
