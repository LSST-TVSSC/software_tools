# Configuration functions relevant to the Nancy Grace Roman Space Telescope
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

class RomanGalBulgeSurvey():
    """
    Configuration of the Nancy Grace Roman Space Telescope observing seasons
    in the Roman Galactic Bulge Time Domain Survey
    """

    def __init__(self):

        # Center of the RGBTDS survey footprint
        self.bulge_field = SkyCoord('18:02:48.75', '-35:41:07.20',
                                    frame='icrs', unit=(u.hourangle, u.deg))
        nominal_seasons = [
                {'start': '2026-02-12T00:00:00', 'end': '2026-04-24T00:00:00'},
                {'start': '2026-09-19T00:00:00', 'end': '2026-10-29T00:00:00'},
                {'start': '2027-02-12T00:00:00', 'end': '2027-04-24T00:00:00'},
                {'start': '2027-09-19T00:00:00', 'end': '2027-10-29T00:00:00'},
                {'start': '2028-02-12T00:00:00', 'end': '2028-04-24T00:00:00'},
                {'start': '2028-09-19T00:00:00', 'end': '2028-10-29T00:00:00'},
                ]

        self.seasons = []
        self.season_gaps = []
        for i,season in enumerate(nominal_seasons):
            tstart = Time(season['start'], format='isot')
            tend = Time(season['end'], format='isot')
            self.seasons.append( {'start': tstart, 'end': tend} )
            if i>=1:
                gap_start = self.seasons[i-1]['end']
                gap_end = self.seasons[i]['start']
                self.season_gaps.append( {'start': gap_start, 'end': gap_end})
