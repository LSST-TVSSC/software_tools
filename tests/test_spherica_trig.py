from os import getcwd, path
from sys import path as pythonpath
from astropy import units as u
from astropy.coordinates import Galactic, SkyCoord, TETE
from astropy_healpix import HEALPix
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

code_dir = path.join(getcwd(), '../')
pythonpath.append(code_dir)

import generate_sky_maps

LMC_params = {'l_center': 280.4652, 'b_center': -32.888443,
          'l_width': 322.827/60, 'b_height': 274.770/60}
SMC_params = {'l_center': 302.8084, 'b_center': -44.3277,
          'l_width': 158.113/60, 'b_height': 93.105/60}

# Object included to test code around the Galactic pole; radius increased
# for the purposes of the test
NGC288_params = {'l_center': 151.285, 'b_center': -89.38,
          'l_width': 5, 'b_height': 5}

# Another stress test, this time for the Galactic center:
GalCenter_params = {'l_center': 0.0, 'b_center': 0.0,
          'l_width': 20.0, 'b_height': 20.0}

# Worst case scenario:
patho_params = {'l_center': 0.0, 'b_center': -89.0,
          'l_width': 20.0, 'b_height': 20.0}

NSIDE = 64

def test_calc_box_corners():

    for target_params in [LMC_params, SMC_params]:
        method1_results = calc_box_corners_original(target_params)
        method2_results = calc_box_corners_astropy(target_params)

        np.testing.assert_almost_equal(method1_results, method2_results, 3)

def test_select_points_within_radius():

    corners = calc_box_corners_astropy(LMC_params)

    n_points = 4
    l = np.linspace(corners[0], corners[1], n_points) * u.deg
    b = np.linspace(corners[2], corners[3], n_points) * u.deg

    LL,BB = np.meshgrid(l, b)
    coords = np.stack((LL.flatten(),BB.flatten()), axis=1)
    pointings = SkyCoord(coords[:,0], coords[:,1],frame=Galactic())

    coords_method1 = calc_points_in_circle_original(LMC_params,coords)
    coords_method2 = calc_points_in_circle_astropy(LMC_params,pointings)

    sep = coords_method1.separation(coords_method2)

    assert (sep < 2*u.arcsec).all()

def test_calc_healpix_for_region():

    ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())
    ahp_test = HEALPix(nside=NSIDE, order='ring', frame=TETE())

    for target_params in [LMC_params, SMC_params]:
        r = generate_sky_maps.CelestialRegion(target_params)
        r.calc_healpixels_for_region(ahp)
        region_pixels = np.unique(r.pixels)

        corners = calc_box_corners_astropy(target_params)

        n_points = 500
        l = np.linspace(corners[0], corners[1], n_points) * u.deg
        b = np.linspace(corners[2], corners[3], n_points) * u.deg

        LL,BB = np.meshgrid(l, b)
        coords = np.stack((LL.flatten(),BB.flatten()), axis=1)
        pointings = SkyCoord(coords[:,0], coords[:,1],frame=Galactic())

        coords_in_region = calc_points_in_circle_original(target_params,coords)

        pixels = ahp_test.skycoord_to_healpix(coords_in_region)
        test_pixels = np.unique(pixels.flatten())

        assert np.all(region_pixels == test_pixels)

def calc_box_corners_original(target_params):

    cosfactor = np.cos(target_params['b_center']*np.pi/180.0)
    cosfactor2 = cosfactor*cosfactor
    halfwidth_l = target_params['l_width'] / 2.0 / cosfactor
    halfheight_b = target_params['b_height'] / 2.0

    l_min = max( (target_params['l_center']-halfwidth_l), 0 )
    l_max = min( (target_params['l_center']+halfwidth_l), 360.0 )
    b_min = max( (target_params['b_center']-halfheight_b), -90.0 )
    b_max = min( (target_params['b_center']+halfheight_b), 90.0 )

    print('Box corners from original method: ',l_min, l_max, b_min, b_max)

    return np.array([l_min, l_max, b_min, b_max])

def calc_box_corners_astropy(target_params):

    target = SkyCoord(target_params['l_center']*u.deg, target_params['b_center']*u.deg,
                        frame=Galactic())

    halfwidth = target_params['l_width']*u.deg / 2.0
    halfheight = target_params['b_height']*u.deg / 2.0

    left_box = target.spherical_offsets_by(-halfwidth, 0*u.deg)
    top_box = target.spherical_offsets_by(0*u.deg, halfheight)
    right_box = target.spherical_offsets_by(halfwidth, 0*u.deg)
    bottom_box = target.spherical_offsets_by(0*u.deg, -halfheight)

    l_min = left_box.l.value
    l_max = right_box.l.value
    b_min = bottom_box.b.value
    b_max = top_box.b.value

    print('Box corners from astropy method: ',l_min, l_max, b_min, b_max)

    return np.array([l_min, l_max, b_min, b_max])

def calc_points_in_circle_original(target_params,coords,use_circles=True):

    if use_circles:
        cosfactor = np.cos(target_params['b_center']*np.pi/180.0)
        cosfactor2 = cosfactor*cosfactor
        halfheight_b = target_params['b_height'] / 2.0*u.deg

        separations = np.sqrt(
        (coords[:,0]-target_params['l_center']*u.deg)*(coords[:,0]-target_params['l_center']*u.deg)*cosfactor2 +
        (coords[:,1]-target_params['b_center']*u.deg)*(coords[:,1]-target_params['b_center']*u.deg))

        pointsincircle = np.where(separations <= halfheight_b)

        coords_in_circle = SkyCoord(coords[pointsincircle,0], coords[pointsincircle,1], frame=Galactic())
    else:
        coords_in_circle = SkyCoord(coords[:,0], coords[:,1], frame=Galactic())

    return coords_in_circle

def calc_points_in_circle_astropy(target_params,pointings,use_circles=True):

    if use_circles:
        center = SkyCoord(target_params['l_center']*u.deg,
                          target_params['b_center']*u.deg,
                            frame=Galactic())

        #threshold = max(target_params['l_width']/2.0* u.deg,
        #                target_params['b_height']/2.0* u.deg)
        threshold = target_params['b_height'] / 2.0 * u.deg

        separations = center.separation(pointings)

        idx = np.where(separations <= threshold)

        coords_in_circle = pointings[idx]
    else:
        coords_in_circle = pointings

    return coords_in_circle

def test_regions():
    NPIX = hp.nside2npix(NSIDE)

    ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())

    #r = generate_sky_maps.CelestialRegion(LMC_params)
    r = generate_sky_maps.CelestialRegion(GalCenter_params)
    #r = generate_sky_maps.CelestialRegion(patho_params)
    #r = generate_sky_maps.CelestialRegion(NGC288_params)
    r.calc_healpixels_for_region(ahp,diagnostics=True)

    map = np.zeros(NPIX)
    map[r.pixels] += 1.0

    fig = plt.figure(1,(10,10))
    hp.mollview(map, title='Test region')
    hp.graticule()
    plt.tight_layout()
    plt.savefig('region_test_plot.png')
    plt.close(1)

def test_calc_lb_ranges():

    # Expected test results
    test_regions = [
                    {'params': LMC_params, 'l_ranges': [(277.774975, 283.155425)],
                                            'b_ranges': [(-35.178193, -30.5986930)]},
                    {'params': NGC288_params, 'l_ranges': [(141.285, 161.285)],
                                            'b_ranges': [(80.62, 90.0), (-79.38, -90.0)]},
                    {'params': GalCenter_params, 'l_ranges': [(0.0, 10.0), (350.0, 360.0)],
                                            'b_ranges': [(-10.0, 10.0)]},
                    {'params': patho_params, 'l_ranges': [(0.0, 10.0), (350.0, 360.0)],
                                            'b_ranges': [(81.0, 90.0), (-79.0, -90.0)]},
    ]

    for region in test_regions:
        r = generate_sky_maps.CelestialRegion(region['params'])
        l_ranges = r.det_range_gal_long()
        b_ranges = r.det_range_gal_lat()

        print('Test l range: ',region['l_ranges'])
        print('Result l range: ',l_ranges)
        print('Test b range: ',region['b_ranges'])
        print('Result b range: ',b_ranges)

        assert len(l_ranges) == len(region['l_ranges'])
        assert len(b_ranges) == len(region['b_ranges'])
        for i,coord_range in enumerate(region['l_ranges']):
            np.testing.assert_almost_equal(l_ranges[i][0].value, coord_range[0], decimal=4)
            np.testing.assert_almost_equal(l_ranges[i][1].value, coord_range[1], decimal=4)
        for i,coord_range in enumerate(region['b_ranges']):
            np.testing.assert_almost_equal(b_ranges[i][0].value, coord_range[0], decimal=4)
            np.testing.assert_almost_equal(b_ranges[i][1].value, coord_range[1], decimal=4)

def test_calc_healpixels_for_region():
    ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())
    NPIX = hp.nside2npix(NSIDE)
    map = np.zeros(NPIX)

    test_regions = [NGC288_params]

    for region in test_regions:
        r = generate_sky_maps.CelestialRegion(region)
        r.calc_healpixels_for_region(ahp)
        map[r.pixels] += 1.0

        # Approximate test: Compare the combined area of the HEALpixels to the
        # minimum (cartesian) area that the region should cover.
        # True area will always be greater.
        min_expected_area = np.pi * (NGC288_params['l_width']*u.deg/2.0)**2
        healpix_area = len(r.pixels) * ahp.pixel_area
        assert healpix_area > min_expected_area.to(u.steradian)

    fig = plt.figure(1,(10,10))
    hp.mollview(map, title='Test of HEALpix regions')
    hp.graticule()
    plt.tight_layout()
    plt.savefig('test_cone_search_regions.png')
    plt.close(1)

if __name__ == '__main__':
    test_calc_healpixels_for_region()
