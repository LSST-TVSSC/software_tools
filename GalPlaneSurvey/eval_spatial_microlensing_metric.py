import os
from sys import argv
from sys import path as pythonpath
pythonpath.append('../../')
pythonpath.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rubin_sim.maf as maf
from rubin_sim.data import get_data_dir
from rubin_sim.data import get_baseline
from rubin_sim.utils import hpid2RaDec, equatorialFromGalactic
import healpy as hp
from astropy import units as u
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
import generate_sky_maps
import subprocess

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
PIXAREA = hp.nside2pixarea(NSIDE,degrees=True)
plot_png = False

seed = 42
np.random.seed(seed)

SCIENCE_MAPS = ['combined_map', 'galactic_plane_map','magellenic_clouds_map',
                'galactic_bulge_map', 'pencilbeams_map']

def run_metrics(params):

    # Load the current OpSim database
    runName = os.path.split(params['opsim_db_file'])[-1].replace('.db', '')
    #opsim_db = maf.OpsimDatabase(params['opsim_db_file'])

    # Simulate a sample of events per HEALpixel, with the same tE, but random
    # ranges for u0 and t0.  The metric is then calculated for all HEALpixels
    # in the sky, and the results coadded in map form
    tE_range = [30.0, 200.0]
    nevents_per_hp = 100
    nproc_in_batch = 10
    t_start=1
    t_end=3652
    processes = []

    for tE in tE_range:
        for i in range(0,nevents_per_hp,1):
            t0 = np.random.uniform(low=t_start, high=t_end, size=1)[0]
            u0 = np.random.uniform(low=0.001, high=0.5, size=1)[0]

            processes.append( [tE, t0, u0] )
    print('Generated simulated event parameters for '+str(len(processes))+' trials')

    for pstart in range(0,len(processes),nproc_in_batch):
        pend = pstart + nproc_in_batch
        current_processes = []
        print('Triggering simulations '+str(pstart)+' to '+str(pend))
        for i in range(pstart, pend, 1):
            (tE, t0, u0) = processes[i]
            print('-> '+str(tE)+' '+str(t0)+' '+str(u0))
            pid = trigger_calc_microlensing_metrics(params, t0, u0, tE)
            current_processes.append(pid)

        exit_codes = [p.wait() for p in current_processes]
        print('-> Processes in batch completed')

    print('Completed all simulation processes')


def trigger_calc_microlensing_metrics(params, t0, u0, tE):
    """Based on a notebook by Peter Yoachim and metric by Natasha Abrams, Markus Hundertmark
    and TVS Microlensing group:
    https://github.com/lsst/rubin_sim_notebooks/blob/main/maf/science/Microlensing%20Metric.ipynb
    """

    command = os.path.join(params['software_dir'],'calc_microlensing_metric.py')
    args = [params['python_exec_path'], command,
            params['opsim_db_file'], params['output_dir'],
            str(t0), str(u0), str(tE)]

    p = subprocess.Popen(args, stdout=subprocess.PIPE)

    return p

def get_args():

    params = {}
    if len(argv) == 1:
        params['opsim_db_file'] = input('Please enter the path to the OpSim database: ')
        params['output_dir'] = input('Please enter the path to the output directory: ')
        params['software_dir'] = input('Please enter the path to the software directory: ')
        params['python_exec_path'] = input('Please enter path to Python: ')
    else:
        params['opsim_db_file'] = argv[1]
        params['output_dir'] = argv[2]
        params['software_dir'] = argv[3]
        params['python_exec_path'] = argv[4]

    return params


if __name__ == '__main__':
    params = get_args()
    run_metrics(params)
