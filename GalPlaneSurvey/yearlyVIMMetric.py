################################################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
from types import MethodType
import numpy as np
from rubin_scheduler.utils.season_utils import calc_season
from rubin_sim.maf.maps.galactic_plane_priority_maps import (
    gp_priority_map_components_to_keys,
)
from rubin_sim.maf.metrics.galactic_plane_metrics import galplane_priority_map_thresholds
from rubin_sim.maf.metrics.season_metrics import find_season_edges
from rubin_sim.maf.metrics.base_metric import BaseMetric

__all__ = [
    "calc_interval_decay",
    "YearlyVIMMetric",
]

TAU_OBS = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])


def calc_interval_decay(delta_tobs, tau):
    # Decay constant for metric value relationship as function of obs interval
    K = 1.0 / tau
    m = np.exp(-K * (delta_tobs - tau))
    # But where observation interval is <= tau, replace with 1
    m[np.where(delta_tobs <= tau)] = 1.0
    return m


# this is a bit of a hack .. it helps us use a variety of tau_obs values,
# and dynamically set up reduce functions
def help_set_reduce_func(obj, metricval, tau):
    def _reduceTau(obj, metricval):
        sum = 0.0
        for iyear in metricval.keys():
            sum += metricval[iyear][tau]
        return sum / len(metricval)

    return _reduceTau


class YearlyVIMMetric(BaseMetric):
    """Evaluate the intervals between sequential observations in a
    lightcurve relative to the scientifically desired sampling interval,
    calculating the results on an annual basis throughout the survey.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column headers contained in the
        priority_GalPlane_footprint_map_data tables.
    tau_obs : `np.ndarray` or `list` of `float`, opt
        Timescales of minimum-required observations intervals for various classes of time variability.
        Default (None), uses TAU_OBS. In general, this should be left as the default and consistent
        across all galactic-plane oriented metrics.
    mag_limit : `float`, opt
        Magnitude limit to use as a cutoff for various observations.
        Default 22.0.
    mjdCol : `str`, opt
        The name of the observation start MJD column. Default 'observationStartMJD'.
    m5Col : `str', opt
        The name of the five sigma depth column. Default 'fiveSigmaDepth'.
    """

    def __init__(
        self,
        science_map,
        tau_obs=None,
        mag_limit=22.0,
        mjdCol="observationStartMJD",
        m5Col="fiveSigmaDepth",
        **kwargs,
    ):
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        # tau_obs is an array of minimum-required observation intervals for
        # four categories of time variability
        if tau_obs is not None:
            self.tau_obs = tau_obs
        else:
            self.tau_obs = TAU_OBS
        # Create reduce functions for the class that are return the metric for each value in tau_obs

        self.mag_limit = mag_limit
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        maps = ["GalacticPlanePriorityMap"]
        if "metricName" not in kwargs:
            metricName = f"YearlyVIMMetric_{self.science_map}"
        else:
            metricName = kwargs["metricName"]
            del kwargs["metricName"]
        for tau in self.tau_obs:
            tauReduceName = f"reduceTau_{tau:.1f}".replace(".", "_")
            newmethod = help_set_reduce_func(self, None, tau)
            setattr(self, tauReduceName, MethodType(newmethod, tauReduceName))
        super().__init__(
            col=[self.mjdCol, self.m5Col], metricName=metricName, maps=maps, **kwargs
        )
        for i, tau in enumerate(self.tau_obs):
            self.reduceOrder[
                f"reduceTau_{tau:.1f}".replace(".", "_").replace("reduce", "")
            ] = i

    def run(self, dataSlice, slicePoint=None):
        # Check if we want to evaluate this part of the sky, or if the weight is below threshold.
        if (
            slicePoint[gp_priority_map_components_to_keys("sum", self.science_map)]
            <= self.priority_map_threshold
        ):
            return self.badval

        # Calculate the metric for each year of the survey:
        metric_data = {}
        survey_start = dataSlice[self.mjdCol].min()
        survey_end = dataSlice[self.mjdCol].max()
        year_start = survey_start
        iyear = 1
        while year_start <= survey_end:
            year_end = year_start + 365.25

            # Select observations in the time sequence that fulfill the
            # S/N requirements:
            idx1 = np.where(dataSlice[self.mjdCol] >= year_start)[0]
            idx2 = np.where(dataSlice[self.mjdCol] < year_end)[0]
            idx3 = np.where(dataSlice[self.m5Col] >= self.mag_limit)[0]
            match = set(idx1).intersection(set(idx2))
            match = list(match.intersection(set(idx3)))

            # We need at least two visits which match these requirements to calculate visit gaps
            if len(match) < 2:
                return self.badval
            # Find the time gaps between visits (in any filter)
            times = dataSlice[self.mjdCol][match]
            times.sort()
            delta_tobs = np.diff(times)
            # Compare the time gap distribution to the time gap required to characterize variability
            annual_metric_data = {}
            for tau in self.tau_obs:
                # Normalize
                annual_metric_data[tau] = calc_interval_decay(delta_tobs, tau).sum() / len(
                    delta_tobs
                )

            metric_data[iyear] = annual_metric_data

            year_start = year_end
            iyear += 1


        return metric_data
