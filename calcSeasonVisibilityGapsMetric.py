################################################################################################
# Metric to evaluate the calcSeasonVisibilityGapsMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import rubin_sim.maf as maf
import healpy as hp
import calcVisitIntervalMetric

class calcSeasonVisibilityGapsMetric(maf.BaseMetric):
    """Metric to evaluate the gap between sequential seasonal gaps in
    observations in a lightcurve relative to the scientifically desired
    sampling interval.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """
    def __init__(self, cols=['fieldRA','observationStartMJD',],
                       metricName='calcSeasonVisibilityGapsMetric',
                       **kwargs):

        """tau_obs is an array of minimum-required observation intervals for four categories
        of time variability"""

        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])
        self.ra_col = 'fieldRA'
        self.mjdCol = 'observationStartMJD'

        super().__init__(col=cols, metricName=metricName, metricDtype='object')

    def calcSeasonGaps(self, dataSlice):
        """Given the RA of a field pointing, and time of observation, calculate the length of
        the gaps between observing seasons.

        Parameters
        ----------
        ra : float
            The RA (in degrees) of the point on the sky
        time : np.ndarray
            The times of the observations, in MJD
        Returns
        -------
        np.ndarray
            Time gaps in days between sequential observing seasons
        """

        seasons = maf.seasonMetrics.calcSeason(dataSlice[self.ra_col], dataSlice[self.mjdCol])
        firstOfSeason, lastOfSeason = maf.seasonMetrics.findSeasonEdges(seasons)
        ngaps = len(firstOfSeason)-1
        season_gaps = dataSlice[self.mjdCol][lastOfSeason[0:ngaps-1]] - dataSlice[self.mjdCol][firstOfSeason[1:ngaps]]

        return season_gaps

    def run(self, dataSlice, slicePoint=None):
        season_gaps = self.calcSeasonGaps(dataSlice)

        # To avoid the intensive calculation of the exact visibility of every pointing
        # for 365d a year, we adopt the pre-calculated values for an example field in
        # the Galactic Bulge, which receives good, but not circumpolar, annual visibility.
        total_time_visible_days = 1975.1256 / 24.0
        expected_gap = 365.24 - total_time_visible_days

        metric_data = TauObsMetricData()
        interval_metric = calcVisitIntervalMetric()
        for i,tau in enumerate(self.tau_obs):
            if tau >= expected_gap:
                metric_data.metric_values[i] =  0.0
                for t in season_gaps:
                    metric_data.metric_values[i] += interval_metric.calc_interval_metric(np.array([t]), tau)
                metric_data.metric_values[i] /= 10.0

            else:
                metric_data.metric_values[i] =  1.0

        return metric_data

    def reduceTau0(self, metric_data):
        return metric_data.metric_values[0]
    def reduceTau1(self, metric_data):
        return metric_data.metric_values[1]
    def reduceTau2(self, metric_data):
        return metric_data.metric_values[2]
    def reduceTau3(self, metric_data):
        return metric_data.metric_values[3]
