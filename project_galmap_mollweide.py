from sys import argv
from os import path
from astropy.io import fits
import healpy as hp
import expore_mollweide
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt

NSIDE = 64

def project_map():

    params = get_args()

    # Calculate scaling for transformation:
    hp_pixscale = hp.max_pixrad(NSIDE,degrees=True)
    params['xmax'] = int(round((360.0/hp_pixscale),0))
    params['ymax'] = int(params['xmax']/2.0)

    # Read in the survey region priority map data, factoring in
    # Rubin's visibility function
    observableMap = fetch_observable_survey_region(params)

    # Calculate the RA,Dec coordinates of all HEALpixel centers in the map
    skyPositions = calc_healpixel_coordinates(observableMap)

    # Calculate the Mollweide x,y positions of each pixel:
    pixelPositions = calc_mollweide_coordinates(skyPositions)

    # Scale the default Mollweide pixel positions to integer pixel locations
    # suitable for output as a FITS image:
    (scaledPositions,params) = scale_mollweide_image_coordinates(params,pixelPositions)

    # Resample the priority map data onto this scaled pixel grid:
    resampledMap = resample_map_data(params,observableMap,scaledPositions)

    # Output resampled image and pixel data:
    output_resampled_data(params, resampledMap, observableMap,
                            skyPositions, pixelPositions, scaledPositions)

def get_args():
    """Read in the FITS format binary table in HEALpixels"""

    params = {}
    if len(argv) == 1:
        params['mapFile'] = input('Please enter the path to the Mollweide galactic map data file: ')
        params['visMapFile'] = input('Please enter the path to the Rubin visibility zone map: ')
        params['suffix'] = input('Please enter the file suffix to use for output (e.g. filter): ')
    else:
        params['mapFile'] = argv[1]
        params['visMapFile'] = argv[2]
        params['suffix'] = argv[3]

    # Add default parameters:
    params['xmax'] = 400.0
    params['ymax'] = params['xmax']/2.0
    params['output_dir'] = path.dirname(params['mapFile'])

    return params

def fetch_observable_survey_region(params):

    with fits.open(params['mapFile']) as hdul:
        mapTable = hdul[1].data

    with fits.open(params['visMapFile']) as hdul1:
        visMapTable = hdul1[1].data

    # Apply visibiltiy restrictions to the priority map data:
    observableMap = mapTable.combined_map * visMapTable.visibility_map

    return observableMap

def calc_healpixel_coordinates(observableMap):
    """
    Calculate the RA, Dec of each HEALpixel center from the list of HEALpixels
    """

    pixelPositions = []
    (theta, phi) = hp.pix2ang(NSIDE,range(0,len(observableMap),1))

    ra = np.rad2deg(phi)
    dec = np.rad2deg((np.pi/2.0) - theta)

    skyPositions = SkyCoord(ra*u.deg,
                             dec*u.deg,
                             frame='icrs')

    return skyPositions

def calc_mollweide_coordinates(skyPositions):

    # Convert the RA,Dec positions to the radial coordinate system used by
    # the Mollweide projection implemented by Healpy:
    (colat,colong) = expore_mollweide.icrsToHPRadialCoords(skyPositions.ra.deg,
                                                           skyPositions.dec.deg)

    # Convert Healpy radial coordinates to Mollweide default radial coordinates.
    # since this conversion is fully reversible:
    (latitude,longitude) = expore_mollweide.HPRadialToMollweideRadialCoords(colat,
                                                                            colong)

    # Convert to Mollweide cartesian coordinates, x,y.  This has to loop due to
    # the need to run the Newton-Raphson approximation separately for each
    # location
    pixelPositions = np.zeros((len(latitude),2))
    for i in range(0,len(latitude),1):
        (x,y) = expore_mollweide.radialToMollweideCoords(latitude[i],longitude[i])
        pixelPositions[i,0] = x
        pixelPositions[i,1] = y

    return pixelPositions

def scale_mollweide_image_coordinates(params,pixelPositions):
    """A FITS image requires a 2D array of values at integer pixel coordinates"""

    xzp = params['xmax']/2.0
    yzp = params['ymax']/2.0
    xgrad = params['xmax']/(pixelPositions[:,0].max()-pixelPositions[:,0].min())
    ygrad = params['ymax']/(pixelPositions[:,1].max()-pixelPositions[:,1].min())

    scaledPositions = np.zeros(pixelPositions.shape, dtype='int')
    for i in range(0,len(pixelPositions),1):
        scaledPositions[i,0] = int(round( (pixelPositions[i,0]*xgrad + xzp), 0))
        scaledPositions[i,1] = int(round( (pixelPositions[i,1]*ygrad + yzp), 0))

    # Store the transformation for later output:
    params['xzp'] = xzp
    params['yzp'] = yzp
    params['xgrad'] = xgrad
    params['ygrad'] = ygrad

    return scaledPositions, params

def resample_map_data(params,observableMap,scaledPositions):

    # For Python image data, the array has to be (y,x):
    resampledMap = np.zeros((scaledPositions[:,1].max()+1,
                             scaledPositions[:,0].max()+1))
    xlimit = scaledPositions[:,0].max()
    ylimit = scaledPositions[:,1].max()

    for i in range(0,len(observableMap),1):
        xc = scaledPositions[i,0]
        xmin = max((xc-3),0)
        xmax = min((xc+3),xlimit)
        yc = scaledPositions[i,1]
        ymin = max((yc-3),0)
        ymax = min((yc+3),ylimit)
        xrange = np.arange(xmin,xmax,1)
        yrange = np.arange(ymin,ymax,1)
        grid = np.meshgrid(yrange,xrange)
        resampledMap[grid] += observableMap[i]

    # Renormalize the priority values to preserve the range
    factor = resampledMap.max() / observableMap.max()
    resampledMap /= factor

    fig = plt.figure(1,(10,10))
    imgplot = plt.imshow(resampledMap)
    plt.title('Galactic science priority map in Mollweide Projection')
    plt.savefig(path.join(params['output_dir'],'resampled_moll_priority_map.png'))
    plt.close(1)

    return resampledMap

def output_resampled_data(params, resampledMap, observableMap,
                            skyPositions, pixelPositions, scaledPositions):

    # File header should contain the pixel transformation function:
    hdr = fits.Header()
    hdr['XZP'] = params['xzp']
    hdr['YZP'] = params['yzp']
    hdr['XGRAD'] = params['xgrad']
    hdr['YGRAD'] = params['ygrad']

    # Build image extension
    image_hdu = fits.PrimaryHDU(resampledMap, header=hdr)

    # Build table extension with HEALpixel information
    c1 = fits.Column(name='HPid',
                    array=range(1,len(skyPositions)+1,1),
                    format='I5')
    c2 = fits.Column(name='RA',
                    array=skyPositions.ra.deg,
                    format='E9.5')
    c3 = fits.Column(name='Dec',
                    array=skyPositions.dec.deg,
                    format='E9.5')
    c4 = fits.Column(name='moll_pixel_x',
                    array=pixelPositions[:,0],
                    format='E8.4')
    c5 = fits.Column(name='moll_pixel_y',
                    array=pixelPositions[:,1],
                    format='E8.4')
    c6 = fits.Column(name='image_pixel_x',
                    array=scaledPositions[:,0],
                    format='I5')
    c7 = fits.Column(name='image_pixel_y',
                    array=scaledPositions[:,1],
                    format='I5')
    c8 = fits.Column(name='priority',
                    array=observableMap,
                    format='E8.4')
    table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8])

    # Output as multi-extension FITS file:
    hdul = fits.HDUList([image_hdu, table_hdu])
    output_file = path.join( path.join(params['output_dir'],
                            'priority_GalPlane_map_mollweide_'+params['suffix']+'.fits') )
    hdul.writeto(output_file, overwrite=True)
    print('Resampled map data output to '+output_file)

def project_map_healpy(params, observableMap):
    """Use Healpy to create a Mollweide projected map array (equal area pixels)"""
    projectedMap = hp.mollview(observableMap,
                                title='Galactic Science Region of Interest',
                                return_projected_map=True)

    # Replace the default inf values for locations outside the map boundaries with
    # None and output the data from this masked array:
    mollMap = projectedMap.data
    mollMap[projectedMap.mask] = None

    # Output projected map as a FITS image:
    new_hdu = fits.PrimaryHDU(mollMap)
    newMapFile = path.join( path.dirname(mapFile), 'priority_GalPlane_map_mollweide.fits')
    new_hdu.writeto(newMapFile, overwrite=True)



if __name__ == '__main__':
    project_map()
