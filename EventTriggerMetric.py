# EventTriggerMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 11/19/2018
# Motivation: Detection of microlensing events (and other transient events) can be detected for follow-up with LSST provided that there are a sufficient number of observations early in the event.
# This metric takes the time scale of the event that detection must occur in (DelMax) and the minimum speration between two observations (DelMin). Encoded in the metric are obs_start and obs_end, which are the first and last all-sky observations included in the OpSim. In a microlensing case, a DelMax on order of one quarter of the time scale of the event would be suitable
# The total LSST observing time is divided into 4*(LSST_mission/DelMax) windows of length DelMax, and returns the fraction of these windows where our detection criteria are satisfied
# Being prepared for presentation in Lund et al (in prep) metrics paper.


from lsst.sims.maf.metrics import BaseMetric
import numpy as np

class EventTriggerMetric(BaseMetric):
   """
   For three observations separated by a minimum time frame, and all occurring within a maximum time frame
   """
   def __init__(self, TimeCol='observationStartMJD', **kwargs):
      self.TimeCol=TimeCol
      self.delmin=kwargs.pop('DelMin', 1)/24. #hours to days
      self.delmax=kwargs.pop('DelMax', 48)/24. #hours to days

      super(EventTriggerMetric, self).__init__(col=[self.TimeCol], **kwargs)

   def run(self, dataSlice, slicePoint=None):
      # start and end of all observations in OpSim run, not just for this point in sky
      obs_start=59580.
      obs_end=63230.
      times=np.asarray(dataSlice[self.TimeCol])
      times=np.sort(times)
      # create start times for all test windows
      testwindows=np.arange(obs_start, obs_end-self.delmax, self.delmin/4.)
      valid=0
      # iterate over all test windows and check if criteria is satisfied
      for timestart in testwindows:
         time_trim=times[(times >= timestart) & (times <= timestart+self.delmax)]
         if time_trim.size > 3:
            time_trim2=time_trim[(time_trim >= time_trim[0]+self.delmin) & (time_trim <= time_trim[-1]-self.delmin)]
            if time_trim2.size > 1:
               valid=valid+1

      #print valid, testwindows.size
      return valid/(1.*testwindows.size)

   def reduceFraction(self, data):
      return data

