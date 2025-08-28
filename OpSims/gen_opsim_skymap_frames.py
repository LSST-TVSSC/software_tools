from os import path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import healpy as hp
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mw_plot import MWSkyMap
import sqlite3
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, UTC

# CONFIGURATION
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
DATA_DIR = '/home/rstreet/rubin_sim_data/sim_baseline'
DB_FILE = 'baseline_v4.3.1_10yrs.db'
OUTPUT_DIR = '/data/LSST/SCOC/OpSims/frames'
# END CONFIG

# Parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('start_frame', help='Index of starting frame', type=int)
parser.add_argument('end_frame', help='Index of last frame', type=int)
args = parser.parse_args()

# Plot colors for the different filters
pcolors = {
    'u': '#6202B0',
    'g': '#02B059',
    'r': '#C48D02',
    'i': '#C45602',
    'z': '#C40802',
    'y': '#630300'
}

# Load OpSim database
ops_db_path = path.join(DATA_DIR, DB_FILE)
run_name = path.split(path.basename(ops_db_path))[-1].replace('.db', '')

# Extract a list of the simulated observations - when they were taken, where in the sky and with what filter
print('Loading set of observations from opsim database...')
conn = sqlite3.connect(ops_db_path)
obs_table = pd.read_sql_query("SELECT * from observations", conn)
conn.close()

# Ensure the observations table is in date-sorted order
obs_table = obs_table.sort_values(by=['observationStartMJD'])

# Determine range of entries to plot per frame
max_mjd = obs_table['observationStartMJD'][0] + 11*365.25

entries = np.where(obs_table['observationStartMJD'] <= max_mjd)[0]

ndp_per_frame = 551     # Approximate number of visits per night
max_frames = int(len(entries) / ndp_per_frame)
if args.start_frame < 0:
    args.start_frame = 0
if args.end_frame > max_frames:
    args.end_frame = max_frames

print('Using number of observations ' + str(len(entries)))
print('Maximum number of frames ' + str(max_frames))
print('Generating frames from ' + str(args.start_frame) + ' to ' + str(args.end_frame))

t1 = datetime.now(UTC)

# Create the background of the animated scene - an all-sky plot of the Milky Way
for i in range(args.start_frame, args.end_frame+1, 1):

    mw1 = MWSkyMap(projection='aitoff', grayscale=False, grid='galactic', background='optical', figsize=(16,10))
    mw1.initialize_mwplot()
    plt.rcParams.update({'font.size': 20})
    proj = HEALPix(nside=NSIDE, order='ring', frame='icrs')

    # The shading will be scaled according to the number of visits per HEALpixel.
    # Histogram the total number of visits in the table per HP to set the scaling
    # normalization range
    field_coords = SkyCoord(obs_table['fieldRA'], obs_table['fieldDec'], frame='icrs', unit=(u.deg, u.deg))
    field_pixels = proj.skycoord_to_healpix(field_coords)
    hist, bin_edges = np.histogram(field_pixels, bins=np.arange(0, NPIX+1, 1))
    norm = mpl.colors.LogNorm(vmin=1.0, vmax=hist.max())

    kmin = 0
    k = kmin + i * ndp_per_frame

    # After the first observation,
    # select all previous observations to date except the most recent so they can
    # be plotted with a lower alpha value.
    # These are plotted with a single color to distinguish them from the most recent obs
    if i > 0:
        # Extract the positions of observations from the table and calculate the HEALpixel
        # indices for each visit to the date of the frame
        s = field_coords[:k-1]
        pixels = field_pixels[:k-1]

        # Histogram the list of pixels to calculate the number of visits per HP
        hist, bin_edges = np.histogram(pixels, bins=np.arange(0, NPIX+1, 1))

        mw1.scatter(
            s.ra.deg * u.deg,
            s.dec.deg * u.deg,
            c=hist[pixels],
            cmap='YlGnBu',
            norm=norm,
            s=40,
            alpha=0.03
        )

    # Plot the most recent observations, looping to apply the appropriate filter color
    kk = max(0, k-ndp_per_frame)
    #for f in range(kk, k, 1):
    s = field_coords[kk:k]
    cols = [pcolors[x] for x in obs_table['band'][kk:k]]
    mw1.scatter(
        s.ra.deg * u.deg,
        s.dec.deg * u.deg,
        c=cols,
        s=40,
        alpha=1.0
    )

    idx = str(i)
    while len(idx) < len(str(max_frames)):
        idx = '0' + idx

    plt.savefig(path.join(OUTPUT_DIR, 'frame_' + idx + '.png'))
    plt.close()

    t2 = datetime.now(UTC)
    dt = t2 - t1
    print('Frame ' + str(i) + ', k='+str(k) + ' completed in ' + str(dt.total_seconds()) + 's')
    t1 = t2
