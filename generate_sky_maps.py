import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from os import path, mkdir
from sys import argv, exit
from astropy import units as u
#from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
from pylab import cm
import csv
import config_utils
import json

# Configuration
NSIDE = 64

class CelestialRegion:
    """Class to describe a region on sky, including its position and
    extend in an on-sky visualization"""

    def __init__(self,params={}):
        self.l_center = None
        self.b_center = None
        self.l_width = None
        self.b_height = None
        self.predefined_pixels = False
        self.pixel_priority = None
        self.NSIDE = NSIDE
        self.NPIX = hp.nside2npix(NSIDE)

        for key, value in params.items():
            if key in dir(self):
                setattr(self,key,value)

        if self.l_width:
            self.halfwidth = self.l_width*u.deg / 2.0
        if self.b_height:
            self.halfheight = self.b_height*u.deg / 2.0

    def calc_ap_healpixels_for_region(self,ahp):

        if not self.predefined_pixels:
            self.skycoord = SkyCoord(self.l_center*u.deg,
                                     self.b_center*u.deg,
                                     frame=Galactic())
            self.pixels = ahp.cone_search_skycoord(self.skycoord,
                                                    self.halfwidth)

    def calc_hp_healpixels_for_region(self):
        """Method calculates the HEALpixels included within the region.
        If the radius of the region is smaller than half that of the HEALpixel map
        resolution, then a minimum radius of 1 HEALpixel is imposed"""

        if not self.predefined_pixels:
            self.skycoord = SkyCoord(self.l_center*u.deg,
                                     self.b_center*u.deg,
                                     frame=Galactic())
            self.skycoord = self.skycoord.transform_to('icrs')
            phi = np.deg2rad(self.skycoord.ra.deg)
            theta = (np.pi/2.0) - np.deg2rad(self.skycoord.dec.deg)
            radius = max(np.deg2rad(self.halfwidth.data),
                         np.deg2rad(hp.max_pixrad(NSIDE,degrees=True)/2.0))
            xyz = hp.ang2vec(theta, phi)
            self.pixels = hp.query_disc(self.NSIDE, xyz, radius)

def build_list_of_regions(target_list):

    regions = []
    for target in target_list:
        params = {'l_center': target['l_center'], 'b_center': target['b_center'],
                  'l_width': target['radius']*2.0, 'b_height': target['radius']*2.0}
        r = CelestialRegion(params)
        r.calc_hp_healpixels_for_region()

        if len(r.pixels) > 0:
            r.pixel_priority = np.zeros(r.NPIX)
            r.pixel_priority[r.pixels] = 1.0
            r.predefined_pixels = True

            regions.append(r)

    return regions

def generate_map():
    """Function to plot given pixel positions on a HEALpix map"""

    # Get user optional parameters
    options = get_args()

    # Fetch configuration:
    config = config_utils.read_config(options['config_file'])

    # Initialze the HEALpix map
    #ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())

    # Load data on the locations of regions of interest depending on user selections:
    maps = {}
    if options['object_list'] == 'O':
        regions = load_open_cluster_data(config)
    elif options['object_list'] == 'G':
        regions = load_globular_cluster_data(config)
    elif options['object_list'] == 'M':
        regions = load_Magellenic_Cloud_data()
    elif options['object_list'] == 'GB':
        regions = load_Galactic_Bulge_data(config)
    elif options['object_list'] == 'GP':
        regions = load_Galactic_Plane_data(config)
    elif options['object_list'] == 'C':
        regions = load_Clementini_region_data(config)
    elif options['object_list'] == 'B':
        regions = load_Bonito_SFR_data(config)
    elif options['object_list'] == 'Z':
        regions = load_SFR_data(config)
    elif options['object_list'] == 'P':
        regions = load_optimized_pencilbeams()
    elif options['object_list'] == 'LP':
        regions = load_larger_pencilbeams()
    elif options['object_list'] == 'K2':
        regions = load_k2_fields(config)
    elif options['object_list'] == 'X':
        regions = None
        maps = load_xray_binary_map(config)
    else:
        raise IOError('Invalid option selected')

    # Build a map combining the pixel regions of all regions of interest:
    if config[options['object_list']]['build_map']:
        maps = build_sky_map(config,options,regions)

    # Output the map data:
    output_sky_map(config,maps,options)

def output_sky_map(config,maps,options):

    if not path.isdir(config['output_dir']):
        mkdir(config['output_dir'])

    map_title = config[options['object_list']]['name']
    root_name = config[options['object_list']]['file_root_name']

    for filter in config['filter_list']:
        fig = plt.figure(3,(10,10))
        hp.mollview(maps[filter], title=map_title)
        hp.graticule()
        plt.tight_layout()
        plt.savefig(path.join(config['output_dir'],root_name+'_'+str(filter)+'.png'))
        plt.close(3)

        use_hp_maps = True
        if use_hp_maps:
            hp.write_map(path.join(config['output_dir'],root_name+'_'+str(filter)+'.fits'), maps[filter], overwrite=True)
        else:
            hdr = fits.Header()
            hdr['NSIDE'] = config['NSIDE']
            hdr['NPIX'] = hp.nside2npix(config['NSIDE'])
            hdr['MAPTITLE'] = map_title
            phdu = fits.PrimaryHDU(header=hdr)
            c1 = fits.Column(name='HEALpix_values', array=maps[filter], format='E')
            hdu = fits.BinTableHDU.from_columns([c1])
            hdul = fits.HDUList([phdu,hdu])
            hdul.writeto(path.join(config['output_dir'],root_name+'_'+str(filter)+'.fits'), overwrite=True)

        print('Output sky map data to '+config['output_dir']+', files '+root_name+'_'+str(filter)+'.png & .fits')

def build_sky_map(config,options,regions):
    print('Generating sky map...')

    maps = {}
    NPIX = hp.nside2npix(config['NSIDE'])

    for filter in config['filter_list']:

        map = np.zeros(NPIX)

        for r in regions:
            #if r.pixel_priority is None:
            #    map[r.pixels] += 1.0
            #else:
            # Combine the pixels from all selected regions
            map += r.pixel_priority

        # Normalize HEALpixel value over all selected regions
        valid_pixels = np.where(map > 0.0)[0]
        max_value = map[valid_pixels].max()
        map[valid_pixels] /= max_value

        # Factor by map weighting functions
        map_weight = config[options['object_list']]['map_weight']
        filter_weight = config[options['object_list']][filter+'_weight']
        map[valid_pixels] = map[valid_pixels] * map_weight
        map[valid_pixels] = map[valid_pixels]*filter_weight

        #if r.pixel_priority is None:
        #    idx = np.where(map > 0.0)
        #    map[idx] = 1.0

        maps[filter] = map
        print('Produced sky map in '+filter+' band with map weighting '
                +str(map_weight)+' and filter weighting '
                +str(filter_weight))

    return maps

def load_Galactic_Plane_data(config):
    # This dictionary defines as keys a set of thresholds in stellar density,
    # with corresponding values indicating the relative priority of pixels of
    # that density; this ensures lower priority to less-dense star fields.
    density_thresholds = { 0.60: 0.8, 0.70: 0.9, 0.80: 1.0 }

    # Load stellar density data from a galactic model datafile
    star_density_map = load_star_density_data(config,limiting_mag=27.5)
    hp_star_density = rotateHealpix(star_density_map)
    idx = hp_star_density > 0.0
    hp_log_star_density = np.zeros(len(hp_star_density))
    hp_log_star_density[idx] = np.log10(hp_star_density[idx])

    # Define boundaries of region
    params = {'l_center': 0.0, 'b_center': 0.0, 'l_width': 170.0, 'b_height': 20.0,
              'u_weight': config['GP']['u_weight'],
              'g_weight': config['GP']['g_weight'],
              'r_weight': config['GP']['r_weight'],
              'i_weight': config['GP']['i_weight'],
              'z_weight': config['GP']['z_weight'],
              'y_weight': config['GP']['y_weight']}
    Gal_Plane = CelestialRegion(params)

    # Assign priority values per HEALpixel based on stellar density
    Gal_Plane.pixel_priority = np.zeros(Gal_Plane.NPIX)
    for threshold, priority in density_thresholds.items():
        idx = np.where(hp_log_star_density >= threshold*hp_log_star_density.max())[0]
        Gal_Plane.pixel_priority[idx] += priority

    # Normalize the range of priority value to zero to 1
    # (Filter weighting is performed in the build_sky_map function)
    max_value = Gal_Plane.pixel_priority.max()
    valid_pixels = np.where(Gal_Plane.pixel_priority > 0.0)[0]
    Gal_Plane.pixel_priority[valid_pixels] /= max_value

    # Necessary to switch off handling of more localized regions, as in this
    # case the pixels are assigned on the basis of stellar density:
    Gal_Plane.predefined_pixels = True
    Gal_Plane.pixels = np.where(Gal_Plane.pixel_priority >0)[0]

    regions = [Gal_Plane]

    return regions

def load_star_density_data(config,limiting_mag=28.0):

    data_file = path.join(config['star_map_dir'], config['GP']['data_file'])
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


def load_open_cluster_data(config):
    """
    Revised to use the prioritized cluster data from Leo Giradi
    This information is distributed into two files, so they must be
    cross-matched and combined for each cluster in priority order.
    """

    pixscale = hp.max_pixrad(NSIDE,degrees=True)

    # Build library of information on clusters from input data files
    # openclusterswithMass.csv file contains the ordered list of clusters with
    # name and location data.  Priority rankings are offset by 1 to avoid
    # dividing by zero later
    clusters = {}
    priority_order = {}
    try:
        file_path = config['O']['data_file'].split()[0]
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i,row in enumerate(reader):
                if i >= 1:
                    line = scan_for_quotes(row[0])
                    entries = line.replace('\n','').split(',')
                    ra = float(entries[2])
                    dec = float(entries[3])
                    s = SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs')
                    s = s.transform_to(Galactic)
                    clusters[entries[1]] = {'l_center': s.l.deg, 'b_center': s.b.deg,
                                'l_width': None, 'b_height': None,
                                'priority': int(float(entries[0]))+1}
                    priority_order[int(float(entries[0]))] = entries[1]

    except:
        raise IOError('Cannot find data files for Open Clusters at '+file_path)

    # Kharchenko_selected.csv file contains required information on cluster
    # radii.  If this is less than the HEALpixel radius, then the map
    # resolution is used to assign a minimum radius to ensure that the
    # cluster is included.
    try:
        file_path = config['O']['data_file'].split()[1]
        with open(file_path, newline='') as csvfile2:
            reader2 = csv.reader(csvfile2, delimiter=' ', quotechar='|')
            for i,row in enumerate(reader2):
                if i >= 1:
                    line = scan_for_quotes(row[0])
                    entries = line.split(',')
                    name = entries[1]
                    if name in clusters.keys():
                        params = clusters[name]
                        params['l_center'] = float(entries[6])
                        params['b_center'] = float(entries[7])
                        lw = max( (float(entries[10])*2.0),pixscale )
                        bh = max( (float(entries[10])*2.0),pixscale )
                        params['l_width'] = lw
                        params['b_height'] = bh

                        clusters[name] = params
    except:
        raise IOError('Cannot find data files for Open Clusters at '+file_path)

    # Assign a priority that scales with the number of clusters,
    # so that priority ranking can be used to assign relative pixel priority
    print('Read in data for '+str(len(clusters))+' open clusters')
    max_priority = 1.0
    n_clusters = float(len(clusters))

    # Now we can build a prioritized list of regions from the list
    regions = []
    for priority,name in priority_order.items():
        params = clusters[name]
        # Exclude any clusters without size information
        if params['l_width'] and params['b_height']:
            r = CelestialRegion(params)

            # Calculate the pixels within the cluster
            r.calc_hp_healpixels_for_region()

            # Some clusters subtend a region smaller than a single HEALpixel
            # These clusters cannot be included in the map
            if len(r.pixels) > 0:
                # Now assign priority to these pixels based on the cluster
                # ranking
                cluster_priority = (-max_priority/(n_clusters+1))*params['priority'] + max_priority
                r.pixel_priority = np.zeros(r.NPIX)
                r.pixel_priority[r.pixels] = cluster_priority
                print('Cluster priority = '+str(params['priority']), cluster_priority)

                # Switch on predefined_pixels to avoid this being reset later
                r.predefined_pixels = True

                regions.append(r)

    print('Loaded data on '+str(len(regions))+' Open Cluster regions')

    return regions

def scan_for_quotes(line):
    if '"' in line:
        i0 = line.index('"')
        i1 = line[i0+1:].index('"')
        new_line = line[0:i0]+line[i0:i1].replace(',','_')+line[i1:]
    else:
        new_line = line
    return new_line

def load_globular_cluster_data(config,use_circles=True):

    pixscale = hp.max_pixrad(NSIDE,degrees=True)
    regions = []
    if path.isfile(config['G']['data_file']):
        with open(config['G']['data_file'], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i,row in enumerate(reader):
                if i >= 1:
                    entries = row[0].split(',')
                    try:
                        if use_circles:
                            # What's the cluster radius we want to sample? Here we set it to 1.5*half-light radius
                            # The half-light radius is r_hl(pc) / R_sun(kpc) converted to deg
                            # However, the radius needs to have a minimum of at least one HEALpixel to register
                            # on the map
                            lw = max( (1.5*(0.001*float(entries[16])/float(entries[5]))*2.0*(180.0/np.pi)),
                                        pixscale )
                            bh = max( (1.5*(0.001*float(entries[16])/float(entries[5]))*2.0*(180.0/np.pi)),
                                        pixscale )
                            params = {'l_center': float(entries[3]), 'b_center': float(entries[4]),
                                      'l_width': lw, 'b_height': bh}
                        else:
                            params = {'l_center': float(entries[3]), 'b_center': float(entries[4]),
                                        'l_width': float(entries[16])*2.0/60.0, 'b_height': float(entries[16])*2.0/60.0}
                        r = CelestialRegion(params)

                        # Calculate the pixels within the cluster
                        r.calc_hp_healpixels_for_region()

                        # Some clusters subtend a region smaller than a single HEALpixel
                        # These clusters cannot be included in the map
                        if len(r.pixels) > 0:
                            # Equal priority is given to all clusters
                            r.pixel_priority = np.zeros(r.NPIX)
                            r.pixel_priority[r.pixels] = 1.0

                            # Switch on predefined_pixels to avoid this being reset later
                            r.predefined_pixels = True

                            regions.append(r)

                    except ValueError:
                        pass
        print('Loaded data on '+str(len(regions))+' Globular Clusters')

    else:
        raise IOError('Cannot find data file for Globular Clusters at '+config['G']['data_file'])

    return regions

def load_Galactic_Bulge_data(config):
    factor = 1.5
    params = {'l_center': 2.216, 'b_center': -3.14,
                'l_width': 3.5*factor, 'b_height': 3.5*factor}
    Bulge = CelestialRegion(params)
    params = {'l_center': 4.67466176, 'b_center': 2.85341714,
                'l_width': 3.5*factor, 'b_height': 3.5*factor}
    NBulge = CelestialRegion(params)

    # Calculate the pixels within these regions and assign equal priority
    regions = []
    for r in [Bulge, NBulge]:
        r.calc_hp_healpixels_for_region()

        r.pixel_priority = np.zeros(r.NPIX)
        r.pixel_priority[r.pixels] = 1.0
        r.predefined_pixels = True

        regions.append(r)
    print('Loaded data on Galactic Bulge')

    return regions

def load_Magellenic_Cloud_data():
    """Basic information taken from SIMBAD
    Factor 2.5 added to radii in both directions for the SMC
    to ensure weighting in clustering algorithm
    """

    LMC_params = {'l_center': 280.4652, 'b_center': -32.888443,
              'l_width': 322.827/60*1.5, 'b_height': 274.770/60*1.5}
    LMC = CelestialRegion(LMC_params)
    SMC_params = {'l_center': 302.8084, 'b_center': -44.3277,
              'l_width': 158.113/60*2.5, 'b_height': 93.105/60*2.5}
    SMC = CelestialRegion(SMC_params)

    # Calculate the pixels within these regions and assign equal priority
    regions = []
    for r in [LMC, SMC]:
        r.calc_hp_healpixels_for_region()

        r.pixel_priority = np.zeros(r.NPIX)
        r.pixel_priority[r.pixels] = 1.0
        r.predefined_pixels = True

        regions.append(r)
    print('Loaded data on Magellenic Clouds')

    return regions


def load_Clementini_region_data(config):
    stellar_populations = [ {'name': 'M54', 'l_center': 5.60703, 'b_center': -14.08715, 'radius': 1.75},
                            {'name': 'Sculptor', 'l_center': 287.5334, 'b_center': -83.1568, 'radius': 1.75},
                            {'name': 'Carina', 'l_center': 260.1124, 'b_center': -22.2235, 'radius': 1.75},
                            {'name': 'Fornax', 'l_center': 237.1038, 'b_center': -65.6515, 'radius': 1.75},
                            {'name': 'Phoenix', 'l_center': 272.1591, 'b_center': -68.9494, 'radius': 1.75} ]

    regions = build_list_of_regions(stellar_populations)

    print('Loaded data on Clementini resolved stellar populations')

    return regions

def load_Bonito_SFR_data(config):
    radius_factor = 1.5
    SFR_regions = [ {'name': 'EtaCarina', 'l_center': 287.5967884538,
                        'b_center': -0.6295111793, 'radius': 1.75*radius_factor},
                    {'name': 'OrionNebula', 'l_center': 209.0137,
                        'b_center': -19.3816, 'radius': 1.75},
                    {'name': 'NGC2264', 'l_center': 202.9358,
                        'b_center': 2.1957, 'radius': 1.75},
                    {'name': 'NGC6530', 'l_center': 6.0828,
                        'b_center': -1.3313, 'radius': 1.75},
                    {'name': 'NGC6611', 'l_center': 16.9540,
                        'b_center': 0.7934, 'radius': 1.75} ]

    regions = build_list_of_regions(SFR_regions)

    print('Loaded data on Bonito Star Forming Regions')

    return regions

def load_SFR_data(config):
    """Function updated to read in Prisinzano table 3 data from
    Low mass young stars in the Milky Way unveiled by DBSCAN and Gaia EDR3. Mapping
    the star forming regions within 1.5 Kpc
        L. Prisinzano, F. Damiani, S. Sciortino, E. Flaccomio, M. G. Guarcello,
        G. Micela, E. Tognelli, R. D. Jeffries, J. M. Alcala'
       <Astron. Astrophys. XXX, XXX (2022)>
       =2022A&A...XXXX        (SIMBAD/NED BibCode)
      See prisinzano_readme for full details.

     The radius provided in the datafile is the r50, including 50% of the
     known members of each object, so this radius is doubled.
     """

    f = open(config['Z']['data_file'], 'r')
    file_lines = f.readlines()
    f.close()

    regions = []
    for line in file_lines:
        if 'name l b' not in line:
            entries = line.replace('\n','').split()
            sfr = {'name': entries[0],
                   'l_center': float(entries[1]),
                   'b_center': float(entries[2]),
                   'l_width': float(entries[3]),
                   'b_height': float(entries[3])}

            # Generate the CelestialRegion object, and use it to
            # calculate the pixels lying within the SFR:
            r = CelestialRegion(sfr)
            r.calc_hp_healpixels_for_region()

            # Now assign priority to these pixels based on 'flag' parameter
            # in the table datafile.
            # This indicates: Flag ([1:28] for SFRs; [29:36] for OCs)
            # Objects with a flag <= 28 are considered to be higher priority
            r.pixel_priority = np.zeros(r.NPIX)
            if float(entries[6]) <= 28.0:
                priority = 1.0
            else:
                priority = 0.5
            r.pixel_priority[r.pixels] += priority

            # Switch on predefined_pixels to avoid this being reset later
            r.predefined_pixels = True

            regions.append(r)

    print('Loaded data on Prisinzano Star Forming Regions')

    return regions

def load_optimized_pencilbeams():
    """Uniformly distributed set of 20 fields placed along the Galactic Plane
    were then optimized in position to maximize the number of stars included.
    This led to a few fields overlapping to the effectively the same position
    in regions with higher extinction.  In these cases, the duplicate was
    removed"""
    pencilbeams_list = [
            {'name': 1, 'l_center': 280.0, 'b_center': 0.0, 'radius': 1.75},
            {'name': 2, 'l_center': 287.280701754386, 'b_center': 0.0, 'radius': 1.75},
            {'name': 3, 'l_center': 295.39473684210526, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 4, 'l_center': 306.42543859649123, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            #{'name': 5, 'l_center': 306.2061403508772, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 6, 'l_center': 320.1535087719298, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 7, 'l_center': 324.51754385964915, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 8, 'l_center': 341.38157894736844, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 9, 'l_center': 351.57894736842104, 'b_center':  -2.5, 'radius': 1.75},
            {'name': 10, 'l_center': 0.10964912280701888, 'b_center':  -2.083333333333333, 'radius': 1.75},
            #{'name': 11, 'l_center': 0.3070175438596484, 'b_center':  -2.083333333333333, 'radius': 1.75},
            {'name': 12, 'l_center': 8.421052631578945, 'b_center':  -3.333333333333333, 'radius': 1.75},
            {'name': 13, 'l_center': 17.36842105263159, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 14, 'l_center': 26.31578947368422, 'b_center':  -2.9166666666666665, 'radius': 1.75},
            {'name': 15, 'l_center': 44.01315789473685, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            #{'name': 16, 'l_center': 44.21052631578948, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 17, 'l_center': 54.40789473684211, 'b_center':  0.0, 'radius': 1.75},
            {'name': 18, 'l_center': 66.27192982456141, 'b_center':  -0.4166666666666661, 'radius': 1.75},
            {'name': 19, 'l_center': 71.8859649122807, 'b_center':  0.0, 'radius': 1.75},
            {'name': 20, 'l_center': 80.0, 'b_center':  -5.0, 'radius': 1.75} ]

    regions = build_list_of_regions(pencilbeams_list)
    print('Loaded data on galactic pencilbeams')

    return regions

def load_larger_pencilbeams():
    pencilbeams_list = [
            {'name': 1, 'l_center': 306.56, 'b_center': -1.46, 'radius': 3.91},
            {'name': 2, 'l_center': 331.09, 'b_center': -2.42, 'radius': 3.91},
            {'name': 3, 'l_center': 18.51, 'b_center': -2.09, 'radius': 3.91},
            {'name': 4, 'l_center': 26.60, 'b_center': -2.15, 'radius': 3.91} ]

    regions = build_list_of_regions(pencilbeams_list)
    print('Loaded data on larger galactic pencilbeams')

    return regions

def load_k2_fields(config):
    f = open(config['K2']['data_file'], 'r')
    k2_fields_data = json.load(f)
    f.close()

    regions = []

    # Start with the main Kepler field.  Couldn't find the CCD-level footprint
    # definition unfortunately
    coord = SkyCoord("19:22:40.0","44:30:00.0", unit=(u.hourangle, u.deg),frame='icrs')
    coord = coord.transform_to('galactic')
    params = {'l_center': float(coord.l.deg), 'b_center': float(coord.b.deg),
                'l_width': 10.7, 'b_height': 10.7}
    r = CelestialRegion(params)
    r.calc_hp_healpixels_for_region()
    r.pixel_priority = np.zeros(r.NPIX)
    r.pixel_priority[r.pixels] = 1.0
    r.predefined_pixels = True

    regions.append(r)

    # Now add each of the K2 fields
    for field_id, field_data in k2_fields_data.items():
        all_pixels = np.array([]).astype('int')

        for channel_id, channel_data in field_data['channels'].items():
            corners = channel_data['corners_ra']
            phi = np.deg2rad(channel_data['corners_ra'])
            theta = (np.pi/2.0) - np.deg2rad(channel_data['corners_dec'])
            xyz = hp.ang2vec(theta, phi)
            pixels = hp.query_polygon(NSIDE, xyz)
            all_pixels = np.concatenate((all_pixels, pixels))

        field_pixels = np.unique(np.array(all_pixels))

        field_center = SkyCoord(field_data['ra'], field_data['dec'],
                                unit=(u.deg,u.deg), frame='icrs')
        field_center = field_center.transform_to('galactic')

        # Exclude fields not close to the Galactic Plane, since this is the
        # focus of this mapping tool
        if abs(field_center.b.deg) < 30.0:
            params = {'l_center': field_center.l.deg, 'b_center': field_center.b.deg,
                      'l_width': 10.7, 'b_height': 10.7}

            r = CelestialRegion(params)
            r.pixel_priority = np.zeros(r.NPIX)
            r.pixel_priority[field_pixels] = 1.0
            r.predefined_pixels = True
            r.pixels = field_pixels

            regions.append(r)

    return regions

def load_xray_binary_map(config):
    """The sky maps of the distribution of x-ray binaries were generated
    by Eric Bellm for each filter, and were pre-weighted according to the
    relative filter balance.

    To ensure these maps are normalized consistently relative to the other
    science maps, this function loads the original maps and scales the
    priority range accordingly.
    """

    # Read in the raw, original map data:
    raw_data = {}
    raw_max_values = {}
    max_value = -1e8
    for f in config['filter_list']:
        file_name = path.join(config['output_dir'],config['X']['input_file_root_name']+'_'+f+'.fits')
        raw_data[f] = hp.read_map(file_name)
        raw_max_values[f] = raw_data[f].max()
        max_value = max( max_value, raw_max_values[f] )
    print(raw_max_values, max_value)

    # Re-normalize and scale the map data
    maps = {}
    for f in config['filter_list']:
        scale_factor = raw_max_values[f] / max_value

        # Re-normalize the map data to the scale used for all science cases
        map = np.zeros(hp.nside2npix(NSIDE))
        valid_pixels = np.where(raw_data[f] > 0.0)[0]
        map[valid_pixels] = (raw_data[f] / raw_max_values[f]) * scale_factor

        # Apply the map-weighting function usually applied
        # during the map build function.  Note no filter weighting is applied
        # here because this is already taken into account in the input maps
        map_weight = config['X']['map_weight']
        filter_weight = config['X'][f+'_weight']
        map[valid_pixels] = map[valid_pixels] * map_weight

        maps[f] = map

    return maps


def get_args():
    """Present menu options and gather user selections"""

    options = {}
    if len(argv) == 1:
        options['config_file'] = input('Please enter the path to the configuration file: ')
        menu = """        Plot Open Cluster locations...........................O
        Plot Globular Cluster locations.......................G
        Plot Magellenic Clouds................................M
        Plot Galactic Bulge...................................GB
        Plot Galactic Plane...................................GP
        Plot Clementini list of resolved stellar populations..C
        Plot Bonito list of Star Forming Regions..............B
        Plot Zucker list of Star Forming Regions..............Z
        Plot pencilbeam fields................................P
        Plot larger pencilbeam fields.........................LP
        Plot K2 fields........................................K2
        Cancel................................................exit"""
        print(menu)
        options['object_list'] = str(input('Please select an option to plot: ')).upper()
    else:
        options['config_file'] = argv[1]
        options['object_list'] = str(argv[2]).upper()

    if 'EXIT' in options['object_list']:
        exit()

    return options


if __name__ == '__main__':
    generate_map()
