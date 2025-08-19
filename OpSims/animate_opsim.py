from os import path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mw_plot import MWSkyMap
import sqlite3
import pandas as pd

# CONFIGURATION
NSIDE = 64
DATA_DIR = '/Users/rstreet/rubin_sim_data/sim_baseline'
DB_FILE = 'baseline_v4.3.1_10yrs.db'
OUTPUT_DIR = '/Users/rstreet/LSST/SCOC/OpSims'
# END CONFIG

# Load OpSim database
ops_db_path = path.join(DATA_DIR, DB_FILE)
run_name = path.split(path.basename(ops_db_path))[-1].replace('.db', '')

# Extract a list of the simulated observations - when they were taken, where in the sky and with what filter
print('Loading set of observations from opsim database...')
conn = sqlite3.connect(ops_db_path)
obs_table = pd.read_sql_query("SELECT * from observations", conn)
conn.close()

# Create the background of the animated scene - an all-sky plot of the Milky Way
print('Creating sky map animation of opsim observations...')
mw1 = MWSkyMap(projection='aitoff', grayscale=False, grid='galactic', background='optical', figsize=(16,10))
mw1.initialize_mwplot()
plt.rcParams.update({'font.size': 20})
proj = HEALPix(nside=NSIDE, order='ring', frame='icrs')

def update(i):
    """
    Frame update function that decides what gets plotted in each frame
    It overlays scatter points to the plot for each observation to date

    Parameters:
        i   int     Frame index
    """
    print('Frame ' + str(i))

    # Plot colors for the different filters
    pcolors = {
        'u': '#6202B0',
        'g': '#02B059',
        'r': '#C48D02',
        'i': '#C45602',
        'z': '#C40802',
        'y': '#630300'
    }

    # After the first observation,
    # select all previous observations to date except the most recent so they can
    # be plotted with a lower alpha value.
    # These are plotted with a single color to distinguish them from the most recent obs
    if i > 0:
        s = SkyCoord(obs_table['fieldRA'][:i-1], obs_table['fieldDec'][:i-1], frame='icrs', unit=(u.deg, u.deg))
        mw1.scatter(
            s.ra.deg * u.deg,
            s.dec.deg * u.deg,
            c='#00BABC',
            s=5,
            alpha=0.03
        )

    # Plot the most recent observation
    s = SkyCoord(obs_table['fieldRA'][i], obs_table['fieldDec'][i], frame='icrs', unit=(u.deg, u.deg))
    mw1.scatter(
        s.ra.deg * u.deg,
        s.dec.deg * u.deg,
        c=pcolors[obs_table['band'][i]],
        s=5,
        alpha=1.0
    )

    return mw1

# Create the animation
ani = animation.FuncAnimation(fig=mw1.fig, func=update, frames=5, interval=30)
#plt.show()
ani.save(filename=path.join(OUTPUT_DIR, DB_FILE.replace('.db', 'mp4')), writer="imagemagick")
