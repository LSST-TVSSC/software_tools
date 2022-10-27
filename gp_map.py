import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from os import path
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
import gc_all_lsst_field

# Configuration
STAR_MAP_DIR = '/Users/rstreet1/software/LSST-TVSSC_software_tools/star_density_maps'
STAR_MAP_FILE = 'TRIstarDensity_r_nside_64.npz'
OUTPUT_DIR = './footprint_maps'

def load_star_density_map(limiting_mag=28.0):

    data_file = path.join(STAR_MAP_DIR, STAR_MAP_FILE)
    if path.isfile(data_file):
        npz_file = np.load(data_file)
        with np.load(data_file) as npz_file:
            star_map = npz_file['starDensity']
            mag_bins = npz_file['bins']

            dmag = abs(mag_bins - limiting_mag)
            idx = np.where(dmag == dmag.min())[0]

            star_density_map = np.copy(star_map[:,idx]).flatten()
            star_density_map = hp.reorder(star_density_map, n2r=True)

        return star_density_map

    else:
        raise IOError('Cannot find star density map data file at '+data_file)

    return None

def rotateHealpix(hpmap, transf=['C','G'], phideg=0., thetadeg=0.):
    """Rotates healpix map from one system to the other. Returns reordered healpy map.
    Healpy coord transformations are used, or you can specify your own angles in degrees.
    To specify your own angles, ensure that transf has length != 2.
    Original code by Xiaolong Li
    """

    # For reasons I don't understand, entering in ['C', 'G'] seems to do the
    # transformation FROM galactic TO equatorial. Possibly something buried in
    # the conventions used by healpy.

    # Heavily influenced by stack overflow solution here:
    # https://stackoverflow.com/questions/24636372/apply-rotation-to-healpix-map-in-healpy

    nside = hp.npix2nside(len(hpmap))

    # Get theta, phi for non-rotated map
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    # Define a rotator
    if len(transf) == 2:
        r = hp.Rotator(coord=transf)
    else:
        r = hp.Rotator(deg=True, rot=[phideg,thetadeg])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hpmap, trot, prot)

    return rot_map

def calc_hp_pixels_for_region(l_center, b_center, l_width, b_height, n_points, ahp):

    halfwidth_l = l_width / 2.0
    halfheight_b = b_height / 2.0

    l_min = max( (l_center-halfwidth_l), 0 )
    l_max = min( (l_center+halfwidth_l), 360.0 )
    b_min = max( (b_center-halfheight_b), -90.0 )
    b_max = min( (b_center+halfheight_b), 90.0 )

    l = np.linspace(l_min, l_max, n_points) * u.deg
    b = np.linspace(b_min, b_max, n_points) * u.deg

    LL,BB = np.meshgrid(l, b)

    coords = SkyCoord(LL, BB, frame=Galactic())

    pixels = ahp.skycoord_to_healpix(coords)

    return pixels

def bono_survey_regions():

    n_points = 500
    l = np.linspace(-20.0, 20.0, n_points) * u.deg
    b = np.linspace(-15.0, 10.0, n_points) * u.deg
    LL,BB = np.meshgrid(l, b)
    coords = SkyCoord(LL, BB, frame=Galactic())
    shallow_pix = ahp.skycoord_to_healpix(coords)


    n_points = 100
    l = np.linspace(-20.0, 20.0, n_points) * u.deg
    b = np.linspace(-3.0, 3.0, n_points) * u.deg
    LL,BB = np.meshgrid(l, b)
    coords = SkyCoord(LL, BB, frame=Galactic())
    deep_pix = ahp.skycoord_to_healpix(coords)

    return shallow_pix, deep_pix

def load_SFR():
    data_file = 'Handbook_Distances_Zucker2020.dat'
    f = open(data_file, 'r')
    file_lines = f.readlines()
    f.close()

    SFR_list = []
    for line in file_lines:
        if 'name l b' not in line:
            entries = line.replace('\n','').split()
            sfr = {'name': entries[0], 'l': float(entries[1]), 'b': float(entries[2])}
            SFR_list.append(sfr)

    return SFR_list

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)

# Trilegal data is in Galactic coordinates, so this needs to be rotated in
# order to map it to healpix
star_density_map = load_star_density_map(limiting_mag=24.7)
hp_star_density = rotateHealpix(star_density_map)
hp_log_star_density = np.log10(hp_star_density)

ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())

# Galactic Plane survey regions, -85.0 < l <+85.0◦, -10.0 < b <+10.0◦
# Street: griz, cadence 2-3d
# Gonzales survey 1: i, N visits over 10yrs
# Gonzales survey 2: grizy, Year 1 only
# Bono shallow: ugrizy 2-3d cadence (WFD)
# Bono deep: izy, 2-3d cadence (WFD)
# Straeder: ugrizy 2-3d cadence or rolling
filterset_gp = { 'u': 0.1, 'g': 0.3, 'r': 0.3, 'i': 0.3, 'z': 0.2, 'y': 0.1 }
gp_region_pix1 = calc_hp_pixels_for_region(43.5, 0.0, 90.0, 20.0, 500, ahp)
gp_region_pix2 = calc_hp_pixels_for_region(317.5, 0.0, 90.0, 20.0, 500, ahp)
gp_region_pix = np.concatenate((gp_region_pix1.flatten(),gp_region_pix2.flatten()))

filterset_Gonzalez_gp = { 'u': 0.0, 'g': 0.0, 'r': 0.0, 'i': 1.0, 'z': 0.0, 'y': 0.0 }
gp_region_pix1 = calc_hp_pixels_for_region(7.5, 0.0, 15.0, 20.0, 500, ahp)
gp_region_pix2 = calc_hp_pixels_for_region(352.5, 0.0, 15.0, 20.0, 500, ahp)
Gonzalez_gp_pix = np.concatenate((gp_region_pix1.flatten(),gp_region_pix2.flatten()))

(Bono_shallow_pix, Bono_deep_pix) = bono_survey_regions()
filterset_bono_shallow = { 'u': 0.1, 'g': 0.1, 'r': 0.2, 'i': 0.2, 'z': 0.2, 'y': 0.2 }
filterset_bono_deep = { 'u': 0.0, 'g': 0.0, 'r': 0.0, 'i': 0.4, 'z': 0.3, 'y': 0.3 }

filterset_Bonito_gp = { 'u': 0.0, 'g': 0.3, 'r': 0.4, 'i': 0.3, 'z': 0.0, 'y': 0.0 }
gp_region_pix1 = calc_hp_pixels_for_region(43.5, 0.0, 90.0, 5.0, 500, ahp)
gp_region_pix2 = calc_hp_pixels_for_region(317.5, 0.0, 90.0, 5.0, 500, ahp)
Bonito_gp_pix = np.concatenate((gp_region_pix1.flatten(),gp_region_pix2.flatten()))

# Magellenic Clouds regions
# Poleski: gri, <1d cadence
# Street: griz, 2-3d cadence
# Clementini: gri, WFD cadence
# Olsen: ugrizy, WFD, logarithmic spacing
# LMC  277.77 - 283.155, -35.17815 - -30.59865
filterset_LMC = { 'u': 0.0, 'g': 0.2, 'r': 0.2, 'i': 0.2, 'z': 0.2, 'y': 0.1 }
LMC_pix = calc_hp_pixels_for_region(280.4652, -32.888443, (322.827/60), (274.770/60), 100, ahp)

# SMC 301.4908 - 304.126, -45.1036 - -43.5518
filterset_SMC = { 'u': 0.0, 'g': 0.2, 'r': 0.2, 'i': 0.2, 'z': 0.2, 'y': 0.1 }
SMC_pix = calc_hp_pixels_for_region(302.8084, -44.3277, (158.113/60), (93.105/60), 100, ahp)

# Street: griz, simultaneous with Rubin + 3d cadence in same years
# Straeder: ugrizy 2-3d cadence or rolling
# Bono shallow: ugrizy 2-3d cadence (WFD)
# Bono deep: izy, 2-3d cadence (WFD)
filterset_bulge = { 'u': 0.1, 'g': 0.2, 'r': 0.3, 'i': 0.3, 'z': 0.2, 'y': 0.2 }
bulge_pix = calc_hp_pixels_for_region(2.216, -3.14, 3.5, 3.5, 50, ahp)

# Clementini survey regions
# gri, WFD cadence
filterset_Clementini = { 'u': 0.0, 'g': 0.3, 'r': 0.4, 'i': 0.3, 'z': 0.0, 'y': 0.0 }
M54_pix = calc_hp_pixels_for_region(5.60703,-14.08715, 3.5, 3.5, 20, ahp)
Sculptor_pix = calc_hp_pixels_for_region(287.5334, -83.1568, 3.5, 3.5, 20, ahp)
Carina_pix = calc_hp_pixels_for_region(260.1124, -22.2235, 3.5, 3.5, 20, ahp)
Fornax_pix = calc_hp_pixels_for_region(237.1038, -65.6515, 3.5, 3.5, 20, ahp)
Phoenix_pix = calc_hp_pixels_for_region(272.1591, -68.9494, 3.5, 3.5, 20, ahp)
Antlia2_pix = calc_hp_pixels_for_region(264.8955, 11.2479, 3.5, 3.5, 20, ahp)
Clementini_regions = np.concatenate((M54_pix.flatten(), Sculptor_pix.flatten()))
for cluster in [Carina_pix, Fornax_pix, Phoenix_pix, Antlia2_pix]:
    Clementini_regions = np.concatenate((Clementini_regions, cluster.flatten()))

# Bonito survey regions - WFD obs valuable here?
# ugrizy in WFD plus additional gri every 30min, 10hrs/night for 7 nights.
filterset_Bonito = { 'u': 0.1, 'g': 0.1, 'r': 0.1, 'i': 0.1, 'z': 0.1, 'y': 0.1 }
EtaCarina_pix = calc_hp_pixels_for_region(287.5967884538, -0.6295111793, 3.5, 3.5, 20, ahp)
OrionNebula_pix = calc_hp_pixels_for_region(209.0137, -19.3816, 3.5, 3.5, 20, ahp)
NGC2264_pix = calc_hp_pixels_for_region(202.9358, 2.1957, 3.5, 3.5, 20, ahp)
NGC6530_pix = calc_hp_pixels_for_region(6.0828, -01.3313, 3.5, 3.5, 20, ahp)
NGC6611_pix = calc_hp_pixels_for_region(16.9540, 0.7934, 3.5, 3.5, 20, ahp)
Bonito_regions = np.concatenate((EtaCarina_pix.flatten(), OrionNebula_pix.flatten()))
for cluster in [NGC2264_pix, NGC6530_pix, NGC6611_pix]:
    Bonito_regions = np.concatenate((Bonito_regions, cluster.flatten()))

# Globular clusters - Figuera Jaimes, Di Stefano
filterset_gc = { 'u': 0.0, 'g': 0.2, 'r': 0.3, 'i': 0.3, 'z': 0.2, 'y': 0.0 }
gc_list = gc_all_lsst_field.fetch_GlobularClusters_in_LSST_footprint()
cluster0_pix = calc_hp_pixels_for_region(gc_list[0]['l'], gc_list[0]['b'], 3.5, 3.5, 20, ahp)
cluster1_pix = calc_hp_pixels_for_region(gc_list[1]['l'], gc_list[1]['b'], 3.5, 3.5, 20, ahp)
gc_regions = np.concatenate((cluster0_pix.flatten(), cluster1_pix.flatten()))
for cluster in gc_list[2:]:
    cluster0_pix = calc_hp_pixels_for_region(cluster['l'], cluster['b'], 3.5, 3.5, 20, ahp)
    gc_regions = np.concatenate((gc_regions, cluster0_pix.flatten()))

# Zucker Star Forming Regions:
filterset_sfr = { 'u': 0.1, 'g': 0.1, 'r': 0.1, 'i': 0.1, 'z': 0.1, 'y': 0.1 }
SFR_list = load_SFR()
sfr0_pix = calc_hp_pixels_for_region(SFR_list[0]['l'], SFR_list[0]['b'], 3.5, 3.5, 20, ahp)
sfr1_pix = calc_hp_pixels_for_region(SFR_list[1]['l'], SFR_list[1]['b'], 3.5, 3.5, 20, ahp)
sfr_regions = np.concatenate((sfr0_pix.flatten(), sfr1_pix.flatten()))
for sfr in SFR_list[2:]:
    sfr0_pix = calc_hp_pixels_for_region(sfr['l'], sfr['b'], 3.5, 3.5, 20, ahp)
    sfr_regions = np.concatenate((sfr_regions, sfr1_pix.flatten()))

# List of high-priority survey regions, in the form of HP pixels:
# Bonito regions as these need an intensive DDF-life strategy
high_priority_regions = {'Galactic_Plane': gp_region_pix,
                         'Gonzalez_Plane_region': Gonzalez_gp_pix,
                         'Bonito_Plane_region': Bonito_gp_pix,
                         'Bono_shallow_survey': Bono_shallow_pix,
                         'Bono_deep_survey': Bono_deep_pix,
                         'Large_Magellenic_Cloud': LMC_pix,
                         'Small_Magellenic_Cloud': SMC_pix,
                         'Galactic_Bulge': bulge_pix,
                         'Clementini_regions': Clementini_regions,
                         'Globular_Clusters': gc_regions,
                         'SFR': sfr_regions}
#                         'Bonito_regions': Bonito_regions}
regions_outside_plane = {'LMC': {'pixel_region': LMC_pix, 'filterset': filterset_LMC},
                         'SMC': {'pixel_region': SMC_pix, 'filterset': filterset_SMC},
                         'Clementini': {'pixel_region': Clementini_regions, 'filterset': filterset_Clementini},
#                         'Bonito': {'pixel_region': Bonito_regions, 'filterset': filterset_Bonito},
                         'Globular_Clusters': {'pixel_region': gc_regions, 'filterset': filterset_gc},
                         'SFR': {'pixel_region': sfr_regions, 'filterset': filterset_sfr},
                         }

# Plotting scale range:
plot_min = 0.598
plot_max = 7.545

# Plot all regions separately for reference:
plotSeparate = False
if plotSeparate:
    for name,region in high_priority_regions.items():
        map = np.zeros(NPIX)
        map[region] = hp_star_density[region]

        fig = plt.figure(1,(10,10))
        hp.mollview(np.log10(map), title=name,
                    min=plot_min, max=plot_max)
        hp.graticule()
        plt.tight_layout()
        plt.savefig(path.join(OUTPUT_DIR,name+'_footprint_map'+'.png'))
        plt.close(1)

# Maximum footprint map:
plotMaxFootprint = False
if plotMaxFootprint:
    max_footprint_map = np.zeros(NPIX)
    for region_name, region_pix in high_priority_regions.items():
        max_footprint_map[region_pix] = hp_star_density[region_pix]

    fig = plt.figure(1,(10,10))
    hp.mollview(np.log10(max_footprint_map), title="Priority regions of the Galactic Plane - maximum footprint",
                min=plot_min, max=plot_max)
    hp.graticule()
    plt.tight_layout()
    plt.savefig(path.join(OUTPUT_DIR,'max_GalPlane_footprint_map.png'))
    plt.close(1)

# Medium footprint map
plotMediumFootprint = False
if plotMediumFootprint:
    density_thresholds = { 0.60: 0.8, 0.70: 0.9, 0.80: 1.0 }
    for f, filter_weight in filterset_gp.items():
        for threshold, location_weight in density_thresholds.items():
            medium_footprint_map = np.zeros(NPIX)
            idx = np.where(hp_log_star_density >= threshold*hp_log_star_density.max())[0]
            medium_footprint_map[idx] = hp_star_density[idx]

            for name, region in regions_outside_plane:
                medium_footprint_map[region['pixel_region']] = hp_star_density[region['pixel_region']]

            plot_min = threshold * plot_max
            fig = plt.figure(2,(10,10))
            hp.mollview(np.log(medium_footprint_map),
                        title="Priority regions of the Galactic Plane - medium footprint, "+\
                            str(round(threshold*100.0,0))+'% of max threshold')
            hp.graticule()
            plt.tight_layout()
            plt.savefig(path.join(OUTPUT_DIR,'medium_GalPlane_footprint_map_'+str(round(threshold*100.0,0))+'.png'))
            plt.close(2)

# Normalize the vote map for the GP by the maximal footprint:
#min_density = np.array(list(density_thresholds.keys())).min()
#threshold = density_thresholds[min_density]
#idx = np.where(hp_log_star_density >= threshold*hp_log_star_density.max())[0]
#vote_maps[idx] = vote_maps[idx]/vote_maps[idx].max()

# Minimum footprint map
plotMinimumFootprint = False
if plotMinimumFootprint:
    minimum_footprint_map = np.zeros(NPIX)
    regions = [Bono_deep_pix,
    LMC_pix, SMC_pix, bulge_pix,
    Clementini_regions, Bonito_regions]
    for region_pix in regions:
        minimum_footprint_map[region_pix] = hp_star_density[region_pix]

    fig = plt.figure(3,(10,10))
    hp.mollview(np.log10(minimum_footprint_map), title="Priority regions of the Galactic Plane - minimum footprint")
    hp.graticule()
    plt.tight_layout()
    plt.savefig(path.join(OUTPUT_DIR,'min_GalPlane_footprint_map.png'))
    plt.close(3)

    plot_min = 0.598
    plot_max = 7.545
    fig = plt.figure(4,(10,10))
    hp.mollview(np.log10(hp_star_density), title="Density of stars within Rubin viewing zone",
                min=plot_min, max=plot_max)
    hp.graticule()
    plt.tight_layout()
    plt.savefig(path.join(OUTPUT_DIR,'rubin_star_density_map.png'))
    plt.close(4)

# Prioritized footprint:
# Add the regions outside the plane with votes equivalent to the number of star
# density thresholds used for the Plane, to give parity.
density_thresholds = { 0.60: 0.8, 0.70: 0.9, 0.80: 1.0 }
vote_maps = {}
for f, filter_weight in filterset_gp.items():
    vote_maps[f] = np.zeros(NPIX)
    for threshold, location_weight in density_thresholds.items():
        idx = np.where(hp_log_star_density >= threshold*hp_log_star_density.max())[0]
        vote_maps[f][idx] += location_weight * filter_weight

for name, region in regions_outside_plane.items():
    for f, filter_weight in region['filterset'].items():
        vote_maps[f][region['pixel_region']] += filter_weight * 1.0


for f in filterset_gp.keys():
    current_max = vote_maps[f].max()*1.0
    #norm = vote_maps[f].max()
    #vote_maps[f] = vote_maps[f]/norm
    fig = plt.figure(3,(10,10))
    hp.mollview(vote_maps[f], title="Priority regions of the Galactic Plane, "+str(f)+"-band",
                min=0.0, max=1.0)
    hp.graticule()
    plt.tight_layout()
    plt.savefig(path.join(OUTPUT_DIR,'priority_GalPlane_footprint_map_'+str(f)+'.png'))
    plt.close(3)

    hp.write_map(path.join(OUTPUT_DIR,'GalPlane_priority_map_'+str(f)+'.fits'), vote_maps[f], overwrite=True)
