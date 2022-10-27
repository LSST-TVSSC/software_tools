import numpy as np
from os import getcwd
from os import path
from sys import argv
import healpy as hp
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
import pyvo
import matplotlib.pyplot as plt
import requests

MAPS_DIR = path.join(getcwd(),'footprint_maps')
NSIDE = 64
TAP_SERVICE_URL = 'https://gaia.ari.uni-heidelberg.de/tap/'

def retrieve_gaia_data_for_map():

    if len(argv) == 1:
        data_dir = input('Please enter the path to the output data directory: ')
    else:
        data_dir = argv[1]

    # Read the Galactic Plane survey priority pointing map, identify those
    # pixels above the given priority threshold, and calculate the spatial
    # coordinates of the corresponding pixels:
    ahp = HEALPix(nside=NSIDE, order='ring', frame=TETE())
    map = hp.read_map(path.join(MAPS_DIR,'GalPlane_priority_map_r.fits'))
    NPIX = hp.nside2npix(NSIDE)
    pixels = np.arange(0,NPIX,1, dtype='int')

    priority_min_threshold = 0.4
    index = (map >= priority_min_threshold,1,0)[0]
    priority_healpixels = np.unique(pixels[index])

    priority_coords = ahp.healpix_to_skycoord(priority_healpixels)
    c = priority_coords[0]

    # For each HEALpix pointing, query the Gaia catalog and retrieve a random
    # sampling of the point source catalog for that sky location:
    tap_service = init_tap_service()

    for i in range(0,len(priority_coords),1):
        coord = priority_coords[i]
        if not check_for_existing_output(data_dir, coord):
            try:
                data = run_async_gaia_tap_query(tap_service, coord)
                output_gaia_to_fits(data_dir,data,coord)
            except requests.exceptions.ConnectionError:
                tap_service = init_tap_service()
                data = run_async_gaia_tap_query(tap_service, coord)
                output_gaia_to_fits(data_dir,data,coord)

def get_gaia_cols():
    return """solution_id, designation, source_id, random_index, ref_epoch, ra, ra_error, dec, dec_error, parallax, parallax_error, parallax_over_error, pm, pmra, pmra_error, pmdec, pmdec_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac, astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al, astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig, astrometric_params_solved, astrometric_primary_flag, nu_eff_used_in_astrometry, pseudocolour, pseudocolour_error, ra_pseudocolour_corr, dec_pseudocolour_corr, parallax_pseudocolour_corr, pmra_pseudocolour_corr, pmdec_pseudocolour_corr, astrometric_matched_transits, visibility_periods_used, astrometric_sigma5d_max, matched_transits, new_matched_transits, matched_transits_removed, ipd_gof_harmonic_amplitude, ipd_gof_harmonic_phase, ipd_frac_multi_peak, ipd_frac_odd_win, ruwe, scan_direction_strength_k1, scan_direction_strength_k2, scan_direction_strength_k3, scan_direction_strength_k4, scan_direction_mean_k1, scan_direction_mean_k2, scan_direction_mean_k3, scan_direction_mean_k4, duplicated_source, phot_g_n_obs, phot_g_mean_flux, phot_g_mean_flux_error, phot_g_mean_flux_over_error, phot_g_mean_mag, phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_mean_flux_over_error, phot_bp_mean_mag, phot_rp_n_obs, phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_mean_flux_over_error, phot_rp_mean_mag, phot_bp_n_contaminated_transits, phot_bp_n_blended_transits, phot_rp_n_contaminated_transits, phot_rp_n_blended_transits, phot_proc_mode, phot_bp_rp_excess_factor, bp_rp, bp_g, g_rp, dr2_radial_velocity, dr2_radial_velocity_error, dr2_rv_nb_transits, dr2_rv_template_teff, dr2_rv_template_logg, dr2_rv_template_fe_h, l, b, ecl_lon, ecl_lat"""

def get_gaia_col_formats():
    return ['D', 'A200'] + ['D']*97

def init_tap_service():
    tap_service = pyvo.dal.TAPService(TAP_SERVICE_URL)
    return tap_service

def run_async_gaia_tap_query(tap_service, coord):
    gaia_cols = get_gaia_cols()

    # Note: Uses the random function to get a random sample of distributions for each location
    query = "SELECT "+gaia_cols+" FROM gaiaedr3.gaia_source WHERE 1=CONTAINS(POINT('', ra, dec), CIRCLE('', "+str(coord.ra.deg)+", "+str(coord.dec.deg)+", 3.5)) AND random_index BETWEEN 5000000 AND 5999999"
    #print(query)

    job = tap_service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
    job.raise_if_error()
    tap_results = job.fetch_result()
    print('Returned '+str(len(tap_results))+' results')

    data = tap_results.to_table()
    return data

def output_gaia_to_fits(data_dir,data,coord):
    gaia_cols = get_gaia_cols()
    gaia_col_formats = get_gaia_col_formats()

    pointing = (coord.to_string('decimal')).replace(' ','_')

    col_list = []
    for i,col_name in enumerate(gaia_cols.split(',')):
        col = fits.Column(name=col_name.strip(), array=data[col_name.strip()], format=gaia_col_formats[i])
        col_list.append(col)
    tbhdu = fits.BinTableHDU.from_columns(col_list)

    header = fits.Header()
    header['NSTARS'] = len(data)
    header['RA'] = coord.ra.deg
    header['DEC'] = coord.dec.deg
    prihdu = fits.PrimaryHDU(header=header)
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(path.join(data_dir,'gaia_edr3_cat_'+pointing+'.fits'), overwrite=True)

def check_for_existing_output(data_dir, coord):
    pointing = (coord.to_string('decimal')).replace(' ','_')
    file_name = path.join(data_dir,'gaia_edr3_cat_'+pointing+'.fits')
    return path.isfile(file_name)

if __name__ == '__main__':
    retrieve_gaia_data_for_map()
