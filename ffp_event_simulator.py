# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:38:27 2018

@author: rstreet
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pyLIMA import microlsimulator, telescopes, microlmodels, event
from pyLIMA import microlfits, microltoolbox
import jplhorizons_utils
import copy

def simulate_ffp():

    spring = True
    fall = False

    u0 = 0.01
    tE = 1.0
    if spring:
        t0 = 2460394.400000
        start_jd = 2460389.500000       # 2024 March 20
        end_jd = 2460399.500000
    if fall:
        t0 = 2460389.500000
        start_jd = 2460573.500000       # 2024 Aug 20
        end_jd = 2460583.500000        # 2024 Aug 30
    ra = 270.0
    dec = -27.0
    piEN = 0.01
    piEE = 0.01
    blend_ratio = 0.2
    source_flux = mag_to_flux(22.0,1.0,24.0)
    blend_flux = source_flux * blend_ratio

    model_params = [ t0, u0, tE, piEN, piEE ]

    lsst_aperture = 6.68
    lsst_read_noise = 10.0

    wfirst_aperture = 2.4
    wfirst_read_noise = 10.0

    if spring:
        horizons_file = 'wfirst_observer_table_spring.txt'
    if fall:
        horizons_file = 'wfirst_observer_table_fall.txt'

    ffp_lsst = event.Event()
    ffp_lsst.name = 'FFP'
    ffp_lsst.ra = ra
    ffp_lsst.dec = dec

    lsst_lc = simulate_lightcurve(start_jd, end_jd, 0.5/24.0, lsst_aperture,
                                  spring, fall, day_gaps=True )
    lsst = telescopes.Telescope(name='LSST', camera_filter='i',
                               location='Earth',
                               light_curve_magnitude=lsst_lc)

    ffp_lsst.telescopes.append(lsst)

    model_lsst = microlmodels.create_model('PSPL', ffp_lsst,
                                      parallax=['Full', t0])
    model_lsst.define_model_parameters()

    fit_lsst = microlfits.MLFits(ffp_lsst)
    fit_lsst.model = model_lsst
    fit_lsst.fit_results = model_params

    ffp_lsst.fits.append(fit_lsst)

    print('Generated event lightcurve from LSST')
    print('Model parameters:')
    print_fit_params(model_params)


    ffp_wfirst = event.Event()
    ffp_wfirst.name = 'FFP'
    ffp_wfirst.ra = ra
    ffp_wfirst.dec = dec

    horizons_table = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path=horizons_file,
                                                                table_type='OBSERVER')

    spacecraft_positions = jplhorizons_utils.extract_spacecraft_positions(horizons_table,t0)

    wfirst_lc = simulate_lightcurve(start_jd, end_jd, 0.25/24.0, wfirst_aperture, spring, fall)

    wfirst = telescopes.Telescope(name='WFIRST', camera_filter='W149',
                               spacecraft_name = 'WFIRST',
                               location='Space',
                               light_curve_magnitude=wfirst_lc)
    wfirst.spacecraft_positions = spacecraft_positions

    ffp_wfirst.telescopes.append(wfirst)

    model_wfirst = microlmodels.create_model('PSPL', ffp_wfirst,
                                      parallax=['Full', t0])

    model_wfirst.define_model_parameters()

    fit_wfirst = microlfits.MLFits(ffp_wfirst)
    fit_wfirst.model = model_wfirst
    fit_wfirst.fit_results = model_params

    ffp_wfirst.fits.append(fit_wfirst)

    print('Generated event lightcurve from WFIRST')
    print('Model parameters:')
    print_fit_params(fit_wfirst)

    parameters = [ t0, u0, tE ]

    lsst_pylima_params = extract_event_parameters(ffp_lsst, fit_lsst, parameters, source_flux, blend_ratio)
    wfirst_pylima_params = extract_event_parameters(ffp_wfirst, fit_wfirst, parameters, source_flux, blend_ratio)

    lsst_lc = add_lensing_event_to_lightcurve(lsst_pylima_params, ffp_lsst,
                                              fit_lsst, lsst_read_noise)

    wfirst_lc = add_lensing_event_to_lightcurve(wfirst_pylima_params, ffp_wfirst,
                                                fit_wfirst, wfirst_read_noise)

    plot_fitted_lightcurves(lsst_lc, wfirst_lc, ffp_lsst, ffp_wfirst,
                            'LSST', 'WFIRST', 'ffp_sim_lightcurve.png',
                            t0=t0,tE=tE)

def extract_event_parameters(lens, fit, event_params, source_flux, blend_ratio):

    parameters = event_params + [ source_flux, blend_ratio ]

    pylima_params = fit.model.compute_pyLIMA_parameters(parameters)

    return pylima_params

def add_lensing_event_to_lightcurve(pylima_params, lens, fit, read_noise):

    A = fit.model.model_magnification(lens.telescopes[0],pylima_params)

    lightcurve = lens.telescopes[0].lightcurve_magnitude

    lightcurve[:,1] = lightcurve[:,1] + -2.5*np.log10(A)

    fluxes = mag_to_flux(lightcurve[:,1], 1.0, 24.0)

    (lightcurve[:,2],read_noise,poisson_noise) = calc_phot_uncertainty(fluxes,
                                                                       read_noise)

    return lightcurve

def simulate_lightcurve(start_jd, end_jd, cadence, aperture, spring, fall,
                        day_gaps=False):
    """Function to generate lightcurve data
    Start and end times in JD, cadence in days^-1
    """

    ts = np.arange(start_jd, end_jd, cadence)

    if day_gaps:

        dec_days = ts % 1

        if spring:

            idx = np.where(dec_days >= 6.4/24.0)[0]
            jdx = np.where(dec_days <= 11.0/24.0)[0]
            kdx = list(set(idx).intersection(set(jdx)))

        if fall:

            idx1 = np.where(dec_days >= 21.5/24.0)[0]
            jdx1 = np.where(dec_days <= 23.9/24.0)[0]
            idx2 = np.where(dec_days >= 0.01/24.0)[0]
            jdx2 = np.where(dec_days <= 3.0/24.0)[0]

            kdx1 = list(set(idx1).intersection(set(jdx1)))
            kdx2 = list(set(idx2).intersection(set(jdx2)))
            kdx = kdx1+kdx2

        ts = ts[kdx]

    lightcurve = np.zeros((len(ts),3))

    lightcurve[:,0] = ts
    lightcurve[:,1] = np.random.normal(loc=18.0,scale=0.02,size=len(ts))

    fluxes = mag_to_flux(lightcurve[:,1], 1.0, 24.0)

    (lightcurve[:,2], read_noise, possion_noise) = calc_phot_uncertainty(fluxes,10.0)

    return lightcurve

def mag_to_flux(mag, gain, ZP):
    """m2 - m1 = -2.5*log(f2/f1)
    f2 = 10**[(m2-m1)/-2.5]"""

    flux = ( 10**( (mag-ZP)/-2.5 ) ) * gain

    return flux

def calc_phot_uncertainty(flux,read_noise):
    """Method to calculate the expected photometric uncertainty for a given
    photometric measurement in flux units.

    :param float mag: Magnitude of star
    """

    aperradius = 3.0

    logfactor = 2.5 * (1.0 / flux) * np.log10(np.exp(1.0))

    npix_aper = np.pi*aperradius*aperradius

    read_noise = np.sqrt(read_noise*read_noise*npix_aper)*logfactor

    possion_noise = np.sqrt(flux)*logfactor

    total_noise = np.sqrt(read_noise*read_noise + possion_noise*possion_noise )

    return total_noise, read_noise, possion_noise

def flux_to_mag(flux, flux_err, gain, ZP):
    """Function to convert the flux of a star from its fitted PSF model
    and its uncertainty onto the magnitude scale.

    :param float flux: Total star flux
    :param float flux_err: Uncertainty in star flux

    Returns:

    :param float mag: Measured star magnitude
    :param float flux_mag: Uncertainty in measured magnitude
    """

    f = flux / gain

    if flux < 0.0 or flux_err < 0.0:

        mag = 0.0
        mag_err = 0.0

    else:

        mag = ZP - 2.5 * np.log10(f)

        mag_err = (2.5/np.log(10.0))*flux_err/f

    return mag, mag_err

def plot_fitted_lightcurves(lc1, lc2, event1, event2, label1, label2, file_path,
                            t0=None, tE=None):
    """Function to plot lightcurves and models"""

    plot_models = False

    dt = float(int(lc1[0,0]))

    ts1 = lc1[:,0] - dt
    ts2 = lc2[:,0] - dt

    fig = plt.figure(1,(10,10))
    plt.subplot(1,1,1)

    plt.plot(ts1,lc1[:,1],marker='.',markersize=10,
             alpha=0.5,color='#8c6931',linestyle='None',
                label=label1)

    plt.plot(ts2,lc2[:,1],marker='.',markersize=10,
             alpha=0.5,color='#2b8c85',linestyle='None',
                label=label2)

    if plot_models:
        model_lc1 = generate_model_lightcurve(event1)

        model_lc2 = generate_model_lightcurve(event2)

        plt.plot(ts1,model_lc1,linestyle='dashed',
                     color='#4c1377')

        plt.plot(ts2,model_lc2,linestyle='solid',
                     color='black')

    plt.xlabel('HJD - '+str(dt), fontsize=24)

    plt.ylabel('Magnitude', fontsize=24)

    plt.legend(loc=1, fontsize=20)

    plt.grid()

    (xmin,xmax,ymin,ymax) = plt.axis()
    if t0 == None:
        plt.axis([xmin,xmax,ymax,ymin])
    else:
        tmin = t0 - dt - 1.5*tE
        tmax = t0 - dt + 1.5*tE
        plt.axis([tmin,tmax,ymax,ymin])

    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)

    plt.savefig(file_path, bbox_inches='tight')

    plt.close(1)

def generate_model_lightcurve(e):
    """Function to produce a model lightcurve based on a parameter set
    fitted by pyLIMA

    Inputs:
    e  Event object

    """

    lc = e.telescopes[0].lightcurve_magnitude

    fit_params = e.fits[-1].model.compute_pyLIMA_parameters(e.fits[-1].fit_results)

    ts = np.linspace(lc[:,0].min(), lc[:,0].max(), len(lc[:,0]))

    reference_telescope = copy.copy(e.fits[-1].event.telescopes[0])

    reference_telescope.lightcurve_magnitude = np.array([ts, [0] * len(ts), [0] * len(ts)]).T

    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()

    if e.fits[-1].model.parallax_model[0] != 'None':

        reference_telescope.compute_parallax(e.fits[-1].event, e.fits[-1].model.parallax_model)

    print_fit_params(fit_params)

    flux_model = e.fits[-1].model.compute_the_microlensing_model(reference_telescope, fit_params)[0]

    mag_model = microltoolbox.flux_to_magnitude(flux_model)

    return mag_model

def print_fit_params(fit_params):

    key_list = ['to', 'uo', 'tE', 'piEN', 'piEE', 'rho',\
                'fs_WFIRST', 'fs_LSST', 'g_WFIRST', 'g_LSST']

    for key in key_list:
        try:
            print(key, getattr(fit_params,key))
        except AttributeError:
            pass

if __name__ == '__main__':

    simulate_ffp()
