import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from os import path, mkdir
from sys import argv, exit
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
from scipy.optimize import curve_fit
import generate_sky_maps
import project_galmap_mollweide

# Configuration
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)

def event_rate_stellar_density(params):
    """Function to approximate the rate of microlensing events as a function
    of stellar density.
    """

    # Read in stellar density data
    config = {'star_map_dir': params['star_map_dir'],
              'GP': {'data_file': params['star_map_file']}}
    star_density_map = generate_sky_maps.load_star_density_data(config,limiting_mag=27.5)
    hp_star_density = generate_sky_maps.rotateHealpix(star_density_map)
    idx = hp_star_density > 0.0
    hp_log_star_density = np.zeros(len(hp_star_density))
    hp_log_star_density[idx] = np.log10(hp_star_density[idx])

    # Create a CelestialRegion object so that we can use the object's methods
    # Select HEALpixels in the central Bulge for which the event rate function
    # is considered to be valid, i.e. +/-10deg around (l,b) = (0,0)
    config = {'l_center': 0.0, 'b_center': 0.0, 'l_width': 20.0, 'b_height': 20.0,
              'u_weight': 1.0, 'g_weight': 1.0, 'r_weight': 1.0,
              'i_weight': 1.0, 'z_weight': 1.0, 'y_weight': 1.0}
    Gal_Centre = generate_sky_maps.CelestialRegion(config)
    Gal_Centre.calc_hp_healpixels_for_region()

    # Calculate the SkyCoords of all HEALpixels, and convert to galactic frame
    pixel_coords = project_galmap_mollweide.calc_healpixel_coordinates(hp_log_star_density)
    pixel_coords = pixel_coords.galactic
    Gal_Centre.pixel_coords = pixel_coords[Gal_Centre.pixels]

    # Calculate the predicted event rate as a function of b for all pixels
    Gal_Centre.gamma = calc_event_rate_for_coords(Gal_Centre.pixel_coords)

    # Extract the stellar density for the selected pixels
    Gal_Centre.log_star_density = hp_log_star_density[Gal_Centre.pixels]

    # For bins of stellar density, select the lowest value of the event rate
    bin_width = 0.05
    density_bins = np.arange(hp_log_star_density[Gal_Centre.pixels].min(),
                             hp_log_star_density[Gal_Centre.pixels].max(),
                             bin_width)
    binned_event_rate = []
    binned_density = []
    for i,bin_min in enumerate(density_bins):
        bin_max = bin_min + bin_width
        idx1 = np.where(Gal_Centre.log_star_density >= bin_min)[0]
        idx2 = np.where(Gal_Centre.log_star_density <= bin_max)[0]
        idx = list(set(idx1).intersection(set(idx2)))
        if len(idx) > 0:
            binned_event_rate.append(Gal_Centre.gamma[idx].min())
            binned_density.append(bin_min)
    binned_density = np.array(binned_density)
    binned_event_rate = np.array(binned_event_rate)

    # Fit a function to the event rate as a function of stellar density
    (pfit, pcov) = curve_fit(power_func, binned_density, binned_event_rate)
    xdata = np.arange(binned_density.min(), binned_density.max(), bin_width)
    ydata = power_func(binned_density, pfit[0], pfit[1], pfit[2], pfit[3])

    # Event rate function, assigning a minimum event rate for HEALpixels
    # where the stellar density is outside the fitted model range.
    event_model = [pfit[0], pfit[1], pfit[2], pfit[3], xdata.min(), ydata.min()]
    print('Fitted model of event rate: ('+str(event_model[0])
            +', '+str(event_model[1])+', '+str(event_model[2])
            +', '+str(event_model[3])
            +') for log stellar density >='+str(event_model[4])+', otherwise '
                +str(event_model[5]))
                
    # Plot event rate as a function of stellar density
    plt.figure(1,(10,10))
    plt.plot(Gal_Centre.log_star_density, Gal_Centre.gamma, 'r.')
    plt.plot(binned_density, binned_event_rate, 'bd')
    plt.plot(xdata, ydata, 'k-')
    plt.xlabel('$log_{10}$(Stellar density) [N deg$^{2}$]')
    plt.ylabel('$\Gamma$ event rate [N yr$^{-1}$]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.join(params['output_dir'],'event_rate_stellar_density.png'))
    plt.close(1)

    plt.figure(2,(10,10))
    xdata = np.arange(0, binned_density.max(), bin_width)
    ydata = event_rate_model(xdata, event_model)
    plt.plot(xdata, ydata, 'k-')
    plt.xlabel('$log_{10}$(Stellar density) [N deg$^{2}$]')
    plt.xlabel('$log_{10}$(Stellar density) [N deg$^{2}$]')
    plt.ylabel('$\Gamma$ event rate [N yr$^{-1}$]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.join(params['output_dir'],'event_rate_stellar_density_function.png'))
    plt.close(2)

def calc_event_rate_for_coords(gal_coords):
    """This function simulates the microlensing rate for each HEALpixel on the
    sky based on the equation of microlensing rate as a function of galactic
    latitude described in Mroz et al (2019) ApJSS, 244, 29.

    No function of galactic longitude is given but it is evident from the figures
    that the rate drops fairly rapidly more than |l| > 10deg away from the central
    Bulge.
    """

    # Extract the array of galactic latitudes for each HEALpixel
    gal_lat = np.array([x.b.deg for x in gal_coords])

    # Calculate the function of microlensing rate
    gamma = 13.4e-6 * np.exp(0.49*(3.0 - abs(gal_lat)))

    return gamma

def generate_map_mulens_rate(params):
    """This function simulates the microlensing rate for each HEALpixel on the
    sky based on the equation of microlensing rate as a function of galactic
    latitude described in Mroz et al (2019) ApJSS, 244, 29.

    No function of galactic longitude is given but it is evident from the figures
    that the rate drops fairly rapidly more than |l| > 10deg away from the central
    Bulge.
    """

    # Calculate the galactic latitude for all HEALpixels in an all-sky map
    gal_coords = calc_healpixel_coordinates(NPIX)

    # Extract the array of galactic latitudes for each HEALpixel
    gal_lat = np.array([x.b.deg for x in gal_coords])

    # Calculate the function of microlensing rate
    gamma = 13.4e-6 * np.exp(0.49*(3.0 - abs(gal_lat)))

    # Estimate the gamma function as a function of l:
    l = [0.0, 1.0, 5.0, 7.5, 10.0]
    gamma = [3.5, 2.5, 1.5, 0.5, 0.2]
    (pfit, pcov) = curve_fit(straight_line_func, l, gamma)
    print(pfit)

    plt.figure(1,(10,10))
    plt.plot(l, gamma, 'r.')
    plt.show()

    # Plot the map
    hp.mollview(gamma, title='Extrapolated map of microlensing rate')
    hp.graticule()
    plt.tight_layout()
    plt.savefig(path.join(params['output_dir'],'mulens_rate_plot.png'))

def straight_line_func(x, a0, a1):
    return a0 + a1*x

def power_func(x, a0, a1, a2, a3):
    data = a0 + a1*x + a2*x**2 + a3*x**3
    idx = np.where(data >= 0.0)[0]
    y = np.zeros(len(x))
    y[idx] = data[idx]
    return y

def event_rate_model(density, event_model):
    y = np.zeros(len(density))
    y.fill(event_model[5])

    idx = np.where(density >= event_model[4])

    data = event_model[0] \
                + event_model[1]*density \
                    + event_model[2]*density**2 \
                        + event_model[3]*density**3

    y[idx] = data[idx]
    return y

def calc_healpixel_coordinates(NPIX):
    """
    Calculate the l,b of each HEALpixel center from the list of HEALpixels
    """

    pixelPositions = []
    (theta, phi) = hp.pix2ang(NSIDE,range(0,NPIX,1))

    ra = np.rad2deg(phi)
    dec = np.rad2deg((np.pi/2.0) - theta)

    skyPositions = SkyCoord(ra*u.deg,
                             dec*u.deg,
                             frame='icrs')
    skyGalactic = skyPositions.transform_to(Galactic)

    return skyGalactic

def get_args():
    params = {}
    if len(argv) == 1:
        params['output_dir'] = input('Please enter the path to the output directory: ')
        params['star_map'] = input('Please enter the path to the stellar density map: ')
        params['']
    else:
        params['output_dir'] = argv[1]
        params['star_map'] = argv[2]

    (params['star_map_dir'], params['star_map_file']) = path.split(params['star_map'])

    return params

if __name__ == '__main__':
    params = get_args()
    event_rate_stellar_density(params)
