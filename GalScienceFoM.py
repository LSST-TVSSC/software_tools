################################################################################################
# Metric to evaluate the Galactic Science Figure of Merit
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric
import healpy as hp
import GalPlaneFootprintMetric, TimePerFilterGalPlane, transientTimeSamplingMetric

class GalScienceFoM(BaseMetric):

    def __init__(self, cols=['fieldRA','fieldDec','filter',
                             'observationStartMJD','visitExposureTime'],
                       metricName='calcVisitIntervalMetric',
                       **kwargs):
        
        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.mjdCol = 'observationStartMJD'
        self.exptCol = 'visitExposureTime'
        
        super(GalScienceFoM,self).__init__(col=cols, metricName=metricName)


    def run(self, dataSlice, slicePoint=None):
        
        footprint_metric = GalPlaneFootprintMetric.GalPlaneFootprintMetric(dataSlice, slicePoint)
        cadence_metric_values = transientTimeSamplingMetric.transientTimeSamplingMetric(dataSlice, slicePoint)
        filters_metric = TimePerFilterGalPlane.GalPlaneTimePerFilter(dataSlice, slicePoint)
        
        # Note: Consider summing cadence_metric_values over all variability categories
        # to produce a single FoM value?
        fom = []
        for t_metric in cadence_metric_values:
            fom.append( footprint_metric*t_metric*filters_metric )
            
        return fom