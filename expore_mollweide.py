import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
import matplotlib.pyplot as plt

NSIDE = 64

def verify_mollweide_conversion():
    test_coordinates = {'bulge': {'l': 2.216, 'b': -3.14, 'symbol': 'D'},
                        'smc': {'l': 302.8084, 'b': -44.3277, 'symbol': '+'},
                        'lmc': {'l': 280.4652, 'b': -32.888443, 'symbol': 'x'}}

    # Forward conversion from Galactic coordinates to Mollweide cartesian
    for name, coords in test_coordinates.items():

        # First convert from galactic coordinates to RA, Dec J2000:
        (coords['ra'],coords['dec']) = galacticToRADec(coords['l'],coords['b'])

        # Now calculate radial coordinates phi and lambda:
        (coords['colatitude'],coords['colongitude']) = icrsToHPRadialCoords(coords['ra'],coords['dec'])

        # Convert Healpy radial coordinates to Mollweide radial coordinates:
        (coords['latitude'],coords['longitude']) = HPRadialToMollweideRadialCoords(coords['colatitude'],coords['colongitude'])

        # Convert to Mollweide cartesian coordinates, x,y:
        (coords['x'],coords['y']) = radialToMollweideCoords(coords['latitude'],coords['longitude'])

        # Scale Mollweide coordinates to match Healpy convention:
        #(coords['xprime'],coords['yprime']) = scaleMollweideCoords(coords['x'],coords['y'])

        # Reverse the transformation:
        (coords['latitude2'],coords['longitude2']) = MollweideToRadialCoords(coords['x'],coords['y'])
        (coords['colatitude2'],coords['colongitude2']) = MollweideRadialToHPRadialCooords(coords['latitude2'],coords['longitude2'])
        (coords['ra2'],coords['dec2']) = HPRadialCoordsToICRS(coords['colatitude2'],coords['colongitude2'])

        # Check we have reacquired the original coordinates:
        np.testing.assert_almost_equal(coords['ra'],coords['ra2'],1e-4)
        np.testing.assert_almost_equal(coords['dec'],coords['dec2'],1e-4)

        print('\n')

    # Plot Mollweide cartesian positions:
    plot_true_moll = True
    fig = plt.figure(1,(10,5))
    for name, coords in test_coordinates.items():
        if plot_true_moll:
            plt.plot(coords['x'],coords['y'],'r', marker=coords['symbol'], markersize=10)
        else:
            plt.plot(coords['xprime'],coords['yprime'],'r', marker=coords['symbol'], markersize=10)
    plt.xlabel('Mollweide x')
    plt.ylabel('Mollweide y')
    if plot_true_moll:
        xmin = -2.0*np.sqrt(2.0)
        xmax = 2.0*np.sqrt(2.0)
        ymin = -np.sqrt(2.0)
        ymax = np.sqrt(2.0)
    else:
        xmin = 0.0
        xmax = 800.0
        ymin = 0.0
        ymax = 400.0
    #plt.axis([xmin,xmax,ymin,ymax])
    print('Mollweide plot ranges: ',xmin,xmax,ymin,ymax)
    plt.savefig('mollweide_forward_conversion.png')
    plt.close(1)

def galacticToRADec(l,b):
    """Convert from galactic coordinates to RA, Dec J2000"""

    s = SkyCoord(l, b, unit=(u.deg,u.deg), frame=Galactic())
    s = s.transform_to('icrs')
    ra = s.ra.degree
    dec = s.dec.degree
    print('Galactic l,b: ',l,b,' -> RA, Dec: ',ra, dec)

    return ra, dec

def icrsToHPRadialCoords(ra,dec,verbose=False):
    """Function assumes input coordinates in decimal degrees, output in radians"""

    colongitude = np.deg2rad(ra)
    colatitude = (np.pi/2.0) - np.deg2rad(dec)
    if verbose:
        print('RA,Dec: ',ra,dec,
                ' -> radial coords co-latitude, co-longitude: ',
                colatitude,colongitude)

    return colatitude, colongitude

def HPRadialCoordsToICRS(colatitude, colongitude):
    """Function input is in radians, output in decimal degrees"""

    ra = np.rad2deg(colongitude)
    dec = np.rad2deg( (np.pi/2.0) - colatitude )
    print('HP Co-latitude, co-longitude: ',colatitude, colongitude,' -> RA, Dec: ',ra,dec)

    return ra,dec

def HPRadialToMollweideRadialCoords(colatitude,colongitude,verbose=False):
    latitude = (np.pi/2.0) - colatitude
    longitude = colongitude

    if verbose:
        print('HP theta, phi: ',colatitude,colongitude,
                ' Mollweide phi,lambda: ',latitude, longitude)

    return latitude, longitude

def MollweideRadialToHPRadialCooords(latitude, longitude, verbose=False):
    colatitude = (np.pi/2.0) - latitude
    colongitude = longitude

    if verbose:
        print('Mollweide phi,lambda: ',latitude, longitude,
                ' HP theta, phi: ',colatitude,colongitude)

    return colatitude, colongitude

def calcTheta(phi):
    """Newton-Raphson approximation of intermediate angle theta"""

    # Catch to avoid division by zero:
    if abs(phi) == np.pi/2:
        return phi

    theta = phi
    stop_criterion = 1e-6
    delta = 1e6

    while delta > stop_criterion:
        newTheta = theta - ( ( (2.0*theta) + (np.sin(2.0*theta)) - (np.pi*np.sin(phi)) ) / (2.0 + 2.0*np.cos(2.0*theta)) )
        delta = abs(theta-newTheta)
        theta = newTheta

    return theta

def radialToMollweideCoords(latitude,longitude,verbose=False):

    theta = calcTheta(latitude)
    if verbose: print('Theta: ',theta)
    R = 1.0     # Radius of projected globe
    longitude0 = 0.0 # Longitude of central meridian

    x = (R * (2.0 * np.sqrt(2.0)) * (longitude-longitude0) * np.cos(theta))/np.pi
    y = R * np.sqrt(2.0) * np.sin(theta)

    if verbose:
        print('Radial phi,lambda: ',latitude,longitude,' -> Mollweide x,y:',x,y)

    return x, y

def scaleMollweideCoords(x,y):
    """Scale the Mollweide coordinates following the Healpy convention
    of xmax=800"""
    R = 1.0     # Radius of projected globe

    xprime = (800.0/(4.0*R*np.sqrt(2.0)))*x + 400
    yprime = (400.0/(2.0*R*np.sqrt(2.0)))*y + 200
    print('Mollweide x,y: ',x,y,' -> Scaled Mollweide xprime,yprime:',xprime,yprime)

    return xprime,yprime

def MollweideToRadialCoords(x,y):
    R = 1.0     # Radius of projected globe
    longitude0 = 0.0 # Longitude of central meridian

    theta = np.arcsin( y/(R*np.sqrt(2.0)) )

    latitude = np.arcsin( (2.0*theta + np.sin(2.0*theta)) / np.pi )
    longitude = longitude0 + ( (np.pi*x) / (2.0*R*np.sqrt(2.0)*np.cos(theta)) )

    print('Reverse Mollweide x,y: ',x,y,' -> radial latitude, longitude: ',latitude, longitude)
    return latitude, longitude

if __name__ == '__main__':
    verify_mollweide_conversion()
