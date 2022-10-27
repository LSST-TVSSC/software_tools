# Example for EventFractionMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 11/19/2018
# Motivation: Some periodic events can be roughly quantified as some fractional duration of an event period, such as the duration of a transit with respect to the period of successive transits.
# This metric for a first order measurement presumes that observations are independently sampled, and determines how likely N number of events will be observed, as determined by the fractional event duration and the total number of observations.
# Sample values for transits of a white dwarf by an Earth-sized planet are 'frac' of order 0.001 and a threshold of 3 points in transit (as in Lund et al 2018). Metric to be discussed in Lund et al (in prep) metrics paper.


from lsst.sims.maf.metrics import BaseMetric
import numpy as np
from scipy.misc import comb

class EventFractionMetric(BaseMetric):
   """
   As a function of number of observations, determine the liklihood that N number of events will be observed.
   """
   def __init__(self, TimeCol='observationStartMJD', **kwargs):
      self.TimeCol=TimeCol
      self.frac=kwargs.pop('frac', 0.1)
      self.event_count=kwargs.pop('event_count', 3.)
      super(EventFractionMetric, self).__init__(col=[self.TimeCol], **kwargs)

   def run(self, dataSlice, slicePoint=None):
      times=np.asarray(dataSlice[self.TimeCol])
      frac=self.frac
      event_count=self.event_count
      num_obs=len(times)
      prob=[]
      # loop interates over the chances of 0 to n-1 observations to calculate the
      # of less than n observations and returns 1-n
      for i in range(self.event_count):
         prob_n=comb(num_obs, i)*frac**i*(1-frac)**(num_obs-i)
         prob.append(prob_n)

      return 1-sum(prob)


