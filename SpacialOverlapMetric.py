# SpacialOverlapMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 11/26/2018
# Motivation: LSST will contribute to science cases of other missions, and it would be useful to understand how much of the sky these overlap for.
# This metric calculates the overlap in the sky between LSST and other surveys or stellar regions by relying on a spherical_geometry python package. An initial set of regions is included (including Stripe 83 and a potential location for WFIRST), but additional regions can be included. Returns the area of overlap.

from lsst.sims.maf.metrics import BaseMetric
import numpy as np
import healpy as hp
from spherical_geometry.polygon import SphericalPolygon
import spherical_geometry.vector as sgv
survey_list={}
survey_pointing_list={}

def box_maker(RA, Dec, width, height):
   hl=height/2.
   #convert decimal degrees dimensions to correct array
   dRA1=width/np.cos((Dec+hl)*np.pi/180)
   dRA2=width/np.cos((Dec-hl)*np.pi/180)
   coord1=np.asarray([RA-dRA2, RA-dRA1, RA+dRA1, RA+dRA2, RA-dRA2]) #RA
   coord2=np.asarray([Dec-hl, Dec+hl, Dec+hl, Dec-hl, Dec-hl]) #Dec
   return coord1, coord2
   

# create the triangle test region 'survey'
temp_survey=[]
coord1=np.asarray([60, 40, 80, 60]) #RA
coord2=np.asarray([10, -20, -20, 10]) #Dec
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(60, -10, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
coord1=np.asarray([160, 140, 180, 160]) #RA
coord2=np.asarray([10, -20, -20, 10]) #Dec
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(160, -10, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
survey_list['triangles']=temp_survey

# create a WFIRST survey area based off of OGLE
temp_survey=[]
coord1=np.asarray([269.3875, 269.3917, 270.7667, 269.3875]) #RA
coord2=np.asarray([-27.9944, -29.2208, -28.6108, -27.9944]) #Dec
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(269.3917, -29.1, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
survey_list['WFIRST']=temp_survey

#VVV survey

# create am OGLE survey area
temp_survey=[]
coord1=np.asarray([251.637, 253.213, 254.792, 256.458, 258.25, 258.4, 257.025, 254.025, 250.937, 247.796, 242.704, 216.833, 209.796, 202.054, 196.35, 190.45, 184.746, 179.287, 169.967, 149.579, 152.15, 154.5, 162.971, 171.992, 174.312, 183.771, 188.567, 193.4, 198.233, 207.825, 214.762, 227.975, 227.912, 237.842, 237.792, 243.267, 246.725, 250.046, 251.637]
) #RA
coord2=np.asarray([-39.456, -38.842, -39.46, -41.311, -43.161, -44.393, -46.24, -48.701, -51.162, -53.623, -56.699, -64.685, -65.296, -65.907, -65.903, -65.903, -65.907, -65.91, -64.685, -58.542, -56.691, -53.609, -56.066, -58.523, -57.905, -59.13, -59.127, -60.355, -59.127, -59.134, -57.291, -54.837, -53.605, -50.533, -49.301, -46.225, -43.764, -41.303, -39.456]) #Dec
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(225, -58, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
coord1=np.asarray([288.325, 280.979, 279.767, 279.817, 283.575, 284.792, 288.496, 288.362, 288.325]) #RA
coord2=np.asarray([4.576, -1.576, -3.422, -10.806, -11.421, -9.575, -7.729, -1.576, 4.576]) #Dec
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(285, -5, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)

survey_list['OGLE_disk']=temp_survey

# create WFIRST pointings
temp_survey=[]
temp_survey.append(sgv.radec_to_vector(269.3875, -27.9944, degrees=True))
temp_survey.append(sgv.radec_to_vector(269.3917, -29.2208, degrees=True))
temp_survey.append(sgv.radec_to_vector(270.7667, -28.6108, degrees=True))
survey_pointing_list['WFIRST']=temp_survey

# create Carina Nebula region
temp_survey=[]
coord1, coord2 = box_maker(161.2875, -59.8678, 2., 2.)
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(161.2875, -59.8678, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
survey_list['Carina']=temp_survey

# create LMC region from http://ned.ipac.caltech.edu/ parameters
temp_survey=[]
coord1, coord2 = box_maker(80.8939, -69.7561, 10.75, 10.75)
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(80.8939, -69.7561, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
survey_list['LMC']=temp_survey

# create Stripe 82 survey
temp_survey=[]
lineset=np.arange(-50,70,10)

coord1=np.asarray([300, 60, 60, 300, 300]) #RA
coord2=np.asarray([-1.26, -1.26, 1.26, 1.26, -1.26]) #Dec
coord1=np.asarray([-50, 59, 59, -50, -50]) #RA
coord2=np.asarray([-1.25, -1.25, 1.25, 1.25, -1.25]) #Dec
coord1=np.asarray([300, 00, 60, 60, 00, 300, 300]) #RA
coord2=np.asarray([-1.26, -1.26, -1.26, 1.26, 1.26, 1.26, -1.26]) #Dec
#coord1=np.append(np.append(lineset, lineset[::-1]), lineset[0])
#coord2=np.append(np.append(np.full(len(lineset),-1.25), np.full(len(lineset[::-1]),1.25)), -1.25)
bounding=np.array(sgv.radec_to_vector(coord1, coord2, degrees=True)).T
inside_val=np.asarray(sgv.radec_to_vector(0, 0, degrees=True))
test_poly=SphericalPolygon(bounding, inside_val)
temp_survey.append(test_poly)
survey_list['stripe82']=temp_survey


class SpacialOverlapMetric(BaseMetric):
   """
   Compare the spacial overlap between LSST and other surveys or sky regions.
   """

   def __init__(self, **kwargs):
      self.region_name=kwargs.pop('region_name')
      super(SpacialOverlapMetric, self).__init__(col=[], units='sr', **kwargs)

   def run(self, dataSlice, slicePoint=None):
      # RA and Dec from dataSlice doesn't necessarily match healpix
      RA=dataSlice['fieldRA'][0]
      Dec=dataSlice['fieldDec'][0]
      # Get RA and Dec from slicer
      RA=slicePoint['ra']
      Dec=slicePoint['dec']
      nside=slicePoint['nside']
      # get the boundaries from HealPix
      bounding=hp.boundaries(nside, slicePoint['sid'], step=1).T
      inside_val=np.asarray(sgv.radec_to_vector(slicePoint['ra'], slicePoint['dec'], degrees=False))
      test_pointing=SphericalPolygon(bounding, inside_val)

      overlap_area=[test_pointing.intersection(survey_polygon).area() for survey_polygon in survey_list[self.region_name]]
      total=sum(overlap_area)
      healpix_area=test_pointing.area()
      return min(total, 4*np.pi-total)
      #designed to correct if wrong region has been selected (interior vs exterior)

