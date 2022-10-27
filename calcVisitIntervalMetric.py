################################################################################################
# Metric to evaluate the calcVisitIntervalMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import rubin_sim.maf as maf
import healpy as hp

class calcVisitIntervalMetric(maf.BaseMetric):
    """Metric to evaluate the intervals between sequential observations in a
    lightcurve relative to the scientifically desired sampling interval.

    Parameters
    ----------
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """

    def __init__(self, cols=['observationStartMJD',],
                       metricName='calcVisitIntervalMetric',
                       **kwargs):
        """tau_obs is an array of minimum-required observation intervals for four categories
        of time variability"""

        self.mjdCol = 'observationStartMJD'
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])

        super().__init__(col=cols, metricName=metricName, metricDtype='object')

    def run(self, dataSlice, slicePoint=None):

        metric_data = TauObsMetricData()

        # Calculate the median time interval from the observation
        # sequence in the dataSlice
        tobs_ordered = dataSlice[self.mjdCol]
        tobs_ordered.sort()
        delta_tobs = tobs_ordered[1:] - tobs_ordered[0:-1]

        # Decay constant for metric value relationship as a function of
        # observation interval
        K = 1.0/self.tau_obs

        for i,tau in enumerate(self.tau_obs):
            m = np.zeros(len(delta_tobs))
            idx = np.where(delta_tobs <= tau)[0]
            m[idx] = 1.0
            idx = np.where(delta_tobs > tau)[0]
            m[idx] = np.exp(-K[i]*(delta_tobs[idx] - tau))

            metric_data.metric_values[i] = m.sum()

            # Normalize by the number of intervals in the lightcurve
            metric_data.metric_values[i] /= len(m)

        return metric_data

    def reduceTau0(self, metric_data):
        return metric_data.metric_values[0]
    def reduceTau1(self, metric_data):
        return metric_data.metric_values[1]
    def reduceTau2(self, metric_data):
        return metric_data.metric_values[2]
    def reduceTau3(self, metric_data):
        return metric_data.metric_values[3]
    
