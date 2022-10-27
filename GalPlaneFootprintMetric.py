#######################################################################################################################
# Metric to evaluate whether a given pointing falls within the region of highest priority for Galactic Plane
# stellar astrophysics
#
# Rachel Street: rstreet@lco.global
#######################################################################################################################
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord

MAP_FILE_ROOT_NAME = 'GalPlane_priority_map'

class GalPlaneFootprintMetric(BaseMetric):

    def __init__(self, cols=['fieldRA','fieldDec','filter'],
                       metricName='GalPlaneFootprintMetric',
                       **kwargs):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.load_maps()

        super(GalPlaneFootprintMetric,self).__init__(col=cols, metricName=metricName)

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(NSIDE)
        for f in self.filters:
            setattr(self, 'map_'+str(f), hp.read_map(MAP_FILE_ROOT_NAME+'_'+str(f)+'.fits')

    def run(self, dataSlice, slicePoint=None):

        coords = SkyCoord(dataSlice[self.ra_col][0], dataSlice[self.dec_col][0], frame=Galactic())

        ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())

        pixels = ahp.skycoord_to_healpix(coords)

        combined_map = np.zeros(self.NPIX)
        for f in self.filters:
            weighted_map = getattr(self, 'map_'+str(f))
            combined_map += weighted_map[pixels]

        metric_value = combined_map.sum()

        return metric_value
        
