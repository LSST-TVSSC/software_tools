import os
from sys import argv
from sys import path as pythonpath
pythonpath.append('/Users/rstreet1/software/rubin_sim_gal_plane/rubin_sim/maf/metrics/')
import numpy as np
import matplotlib.pyplot as plt
import rubin_sim.maf as maf
from rubin_sim.data import get_data_dir
from rubin_sim.data import get_baseline
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
from astropy.time import Time
import galBulgeRubinRomanMetrics

def test_microlensing_event():
    event = galBulgeRubinRomanMetrics.MicrolensingEvent()
    test_t0 = 59000.0
    test_tE = 30.0

    event.t0 = test_t0
    event.tE = test_tE
    event.calcDuration()

    assert event.startMJD == (test_t0 - test_tE)
    assert event.endMJD == (test_t0 + test_tE)

def test_rges_survey_seasons():
    survey = galBulgeRubinRomanMetrics.RGESSurvey()

    for season in survey.seasons:
        assert type(season) == type({})
        assert 'start' in season.keys()
        assert 'end' in season.keys()
        assert type(season['start']) == np.float64
        assert type(season['end']) == np.float64


def test_rges_survey_region():
    survey = galBulgeRubinRomanMetrics.RGESSurvey()
    ahp = HEALPix(nside=64, order='ring', frame=TETE())
    survey.calcHealpix(ahp)

    assert len(survey.pixels > 0)


def test_rges_survey_timestamps():
    survey = galBulgeRubinRomanMetrics.RGESSurvey()

    survey.calcTimeStamps()

    assert type(survey.timestamps) == type(np.array([]))
    assert len(survey.timestamps) > 0

def test_simLensingEvents():
    nSimEvents = 10
    nSeasons = 1
    seasonStart = Time('2026-02-12T00:00:00', format='isot', scale='utc')
    seasonEnd = Time('2026-04-24T00:00:00', format='isot', scale='utc')
    obsSeasons = [{'start': seasonStart.mjd, 'end': seasonEnd.mjd}]

    test_event = galBulgeRubinRomanMetrics.MicrolensingEvent()

    events = galBulgeRubinRomanMetrics.simLensingEvents(nSimEvents,obsSeasons,nSeasons)

    assert len(events) == nSimEvents
    for e in events:
        assert type(e) == type(test_event)
        assert type(e.t0) == type(0.0)
        assert e.t0 > 50000.0
        assert type(e.tE) == type(0.0)
        assert e.tE > 0.0 and e.tE < 400.0
        assert type(e.rho) == type(0.0)
        assert e.rho >= 1e-4 and e.rho <= 0.05

def test_countContemporaneousObs():

    rubinTimestamps = np.linspace(60000.0, 60010.0, 10)
    romanTimestamps = np.linspace(60005.0, 60015.0, 10)

    common_ts = np.array([])
    for ts in romanTimestamps:
        delta = abs(rubinTimestamps - ts)
        idx = np.where(delta <= 1.0)
        common_ts = np.concatenate( (common_ts,rubinTimestamps[idx]) )

    metric = galBulgeRubinRomanMetrics.complementaryObsMetric()

    count = metric.countContemporaneousObs(rubinTimestamps,romanTimestamps)

    assert count == len(common_ts)

def test_countGapObs():

    rubinTimestamps = np.linspace(60000.0, 60020.0, 10)
    romanSeasons = [
                    {'start': 60005.0, 'end': 60015.0},
                    {'start': 60025.0, 'end': 60035.0},
                    ]

    gap_ts = []
    for ts in rubinTimestamps:
        if ts > romanSeasons[0]['end'] and ts < romanSeasons[1]['start']:
            gap_ts.append(ts)

    metric = galBulgeRubinRomanMetrics.complementaryObsMetric()

    count = metric.countGapObs(rubinTimestamps,romanSeasons)

    assert count == len(gap_ts)

if __name__ == '__main__':
    test_countGapObs()
