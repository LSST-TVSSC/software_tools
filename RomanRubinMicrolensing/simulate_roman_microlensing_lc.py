###### Roman Microlensing Lightcurve Simulator
######
###### Code by Etienne Bachelet

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import random
import time


def find_u0_for_caustics(s,q,rho,alpha,origin='central_caustic'):

	aaa = microlcaustics.find_2_lenses_caustics_and_critical_curves(s,q, resolution=1000)

	if origin == 'central_caustic':

		index = 0
	if origin == 'planetary_caustic':

		if s<1:
			index = 1
		else:
			index = 3
	if aaa[0] == 'resonant':
		index = 4


	width = np.max(aaa[1][index].real)-np.min(aaa[1][index].real)
	height = np.max(aaa[1][index].imag)-np.min(aaa[1][index].imag)

	good = np.min([width,height])

	uo = np.random.uniform(-good-1*rho,good+1*rho)
	plt.plot(aaa[1][index].real,aaa[1][index].imag)
	X = -uo*np.sin(alpha)
	Y = uo*np.cos(alpha)
	#patch = plt.Circle((np.median(aaa[1][index].real)+X, np.median(aaa[1][index].imag)+Y), rho,)
	#ax = plt.gca()
	#ax.add_patch(patch)
	#plt.axis('scaled')
	#plt.show()

	return uo

def find_xcaustic_ycaustic(s,q) :

    if s <1 :

        XXX = s-1/(s)
        YYY = 2*(q)**0.5/((s)*(1+(s)**2)**0.5)
        SIGNE = np.random.choice([-1,1])

        return XXX, SIGNE*YYY

    else :

        XXX = s-1/(s)
        YYY = 0
        SIGNE = np.random.choice([-1,1])

        return XXX, SIGNE*YYY

def find_uo_tau(x_caus, y_caus, alpha) :

        magic_y = (np.cos(alpha)*y_caus-np.sin(alpha)*x_caus)/(1)
        magic_x = (x_caus+np.sin(alpha)*magic_y)/(np.cos(alpha))


        return magic_x, magic_y

def find_caustic_size(s,q):
    if s<1 :

        w = q**0.5*s**3

        return np.abs(w)

    else :

        w = 4*q**0.5*(1-1/(2*s**2))

        return np.abs(w)

def find_central_caustic_size(s,q):

    w = 4*q/(s-1/s)**2

    return w



def simulate_a_GL_planet(s_range,q_range):

    to = 2460919
    uo = 8
    delta_t = 165
    caustic_size = 0.0001
    rho = 10
    n_points = 1
    tE = 100
    while ((np.abs(uo)>0.5) | (2460919.5+5>to-np.abs(delta_t)) | (2460919.5+72-5<to+np.abs(delta_t)) | (caustic_size/rho<1.0) | (uo<0.1)) :

        to = np.random.uniform(2460919.5+20,2460919.5+72-20)
        tE = np.random.uniform(5,100)




        rho = np.random.uniform(10**-5,10**-4)

        #print rho
        alpha = np.random.uniform(-np.pi,np.pi)

        MAG = np.random.uniform(18,24)

        log_s = np.random.uniform(s_range[0],s_range[1])
        log_q = np.random.uniform(q_range[0],q_range[1])
        if (log_s < 0) :
            while  np.abs(np.abs(alpha)-np.pi/2)<0.2 :
                alpha = np.random.uniform(-np.pi,np.pi)
        if log_q<-4:

            MAG = np.random.uniform(18,21)
        caustic_size = find_caustic_size(10**log_s,10**log_q)

        XXX,YYY = find_xcaustic_ycaustic(10**log_s,10**log_q)

        #signe = np.random.choice([-1,1])

        #XXX += signe*np.random.uniform(0,1)*caustic_size

        tau, uo = find_uo_tau(XXX, YYY, alpha)


        #tau += np.random.uniform(-0.01,0.01)
        delta_t = tau*tE
        n_points = caustic_size*tE*4*24
        print( caustic_size, rho)
        #print n_points
    print( 'OK')
    return to,uo,tE,rho,log_s,log_q,alpha,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0, delta_t


def simulate_a_NC_planet(s_range,q_range):

    to = 2460919
    uo = 8
    delta_t = 165
    caustic_size = 0.0001
    rho = 10
    n_points = 1
    XXX = 0.0
    while ((np.abs(uo)>0.5) | (2460919.5+5>to-np.abs(delta_t)) | (2460919.5+72-5<to+np.abs(delta_t)) |(uo<0.1) | (caustic_size/rho<1.0)) :

        to = np.random.uniform(2460919.5+20,2460919.5+72-20)
        tE = np.random.uniform(5,100)




        rho = np.random.uniform(5*10**-4,10**-2)
        alpha = np.random.uniform(-np.pi,np.pi)

        MAG = np.random.uniform(18,24)
        log_s = np.random.uniform(s_range[0],s_range[1])
        log_q = np.random.uniform(q_range[0],q_range[1])


        caustic_size = find_caustic_size(10**log_s,10**log_q)

        XXX,YYY = find_xcaustic_ycaustic(10**log_s,10**log_q)





        tau, uo = find_uo_tau(XXX, YYY, alpha)

        uo +=  np.random.choice([1,1.5,2.0])*rho


        delta_t = tau*tE
        n_points = caustic_size*tE*4*24
        #print n_points
        if (XXX<0) & (np.abs(XXX)<uo):

            uo = 5
    return to,uo,tE,rho,log_s,log_q,alpha,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0, delta_t


def simulate_a_HM_planet(s_range,q_range):

    caustic_size = 0.1
    rho = 100
    uo = 0.1
    n_points = 1
    while ((caustic_size/rho<0.1) | (uo/rho>5)):
        to = np.random.uniform(2460919.5+20,2460919.5+72-20)
        tE = np.random.uniform(5,100)
        uo = np.random.uniform(-0.0005,0.01)




        rho = np.random.uniform(-6,-5.0)
        rho = 10**rho
        alpha = np.random.uniform(-np.pi,np.pi)

        MAG = np.random.uniform(18,24)
        log_s = np.random.uniform(s_range[0],s_range[1])
        log_q = np.random.uniform(q_range[0],q_range[1])

        caustic_size = find_central_caustic_size(10**log_s,10**log_q)
        n_points = caustic_size*tE*4*24
        #print caustic_size/rho
        #print caustic_size/rho, n_points, np.abs(uo/rho)
    return to,uo,tE,rho,log_s,log_q,alpha,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0, 0.0

def simulate_a_binary(s_range,q_range):

    to = np.random.uniform(2460919.5+20,2460919.5+72-20)
    tE = np.random.uniform(5,100)

    uo = np.random.uniform(-0.0,0.2)
    delta_t = 0
    alpha = np.random.uniform(-np.pi,np.pi)
    rho = np.random.uniform(-4,-2)
    rho = 10**rho
    MAG = np.random.uniform(18,24)
    log_s = np.random.uniform(s_range[0],s_range[1])
    log_q = np.random.uniform(q_range[0],q_range[1])

    caustic_size = find_central_caustic_size(10**log_s,10**log_q)

    return to,uo,tE,rho,log_s,log_q,alpha,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0, delta_t

def simulate_a_PSPL():


    to = np.random.uniform(2460919.5+20,2460919.5+72-20)
    tE = np.random.uniform(5,100)

    uo = np.random.uniform(-0,1)

    MAG = np.random.uniform(18,24)

    return to,uo,tE,0,0,0,0,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0,10**((27.4-MAG)/2.5), 0.0,0

def generate_name(data):

    name = data.pop()

    return name




ZP = 27.615

from pyLIMA import microlsimulator
from pyLIMA import microlcaustics
from pyLIMA import microlmodels

HEADER = 'Name          Type     to          uo       tE     rho      log_s   log_q   alpha   f_source  g_blending to-time_anomaly \n'


my_own_creation = microlsimulator.simulate_a_microlensing_event(name ='A binary lens observed by WFIRST',
                                                                ra=270, dec=-30)
wfirst1 = microlsimulator.simulate_a_telescope('WFIRST',my_own_creation, 2459000,2459000+72,0.25, 'Space','W149',
                                                  uniform_sampling=True)

wfirst2 = microlsimulator.simulate_a_telescope('WFIRST',my_own_creation, 2459000+365.25,2459000+72+365.25,0.25, 'Space','W149',
                                                  uniform_sampling=True)
wfirst3 = microlsimulator.simulate_a_telescope('WFIRST',my_own_creation, 2459000+2*365.25,2459000+72+2*365.25,0.25, 'Space','W149',
                                                  uniform_sampling=True)

wfirst4 = microlsimulator.simulate_a_telescope('WFIRST',my_own_creation, 2459000+3*365.25,2459000+72+3*365.25,0.25, 'Space','W149',
                                                  uniform_sampling=True)

wfirst5 = microlsimulator.simulate_a_telescope('WFIRST',my_own_creation, 2459000+4*365.25,2459000+72+4*365.25,0.25, 'Space','W149',
                                                  uniform_sampling=True)

wfirst_tot = microlsimulator.simulate_a_telescope('WFIRST',my_own_creation, 2459000+2*365.25,2459000+72+2*365.25,0.25, 'Space','W149',
                                                  uniform_sampling=True)


wfirst_tot.lightcurve_flux = np.r_[wfirst1.lightcurve_flux,wfirst2.lightcurve_flux,wfirst3.lightcurve_flux,wfirst4.lightcurve_flux,wfirst5.lightcurve_flux]
wfirst_tot.gamma = 0.5

my_own_creation.telescopes.append(wfirst_tot)





for each in range(5):
	count = 0
	params = []
	while count<2000:


		#model_choice = np.random.randint(0,5)
		model_choice = each
		t0 = np.random.uniform(np.percentile(wfirst3.lightcurve_flux[:,0],10),np.percentile(wfirst3.lightcurve_flux[:,0],90))
		tE = np.random.normal(30,10)
		rho = np.random.uniform(-4,-1.30)

		rho = 10**rho

		if model_choice == 0:
			directory = 'PSPL'
			u0 = np.random.uniform(-0.5,0.5)
			my_own_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='PSPL')
			my_own_parameters = [t0,u0,tE]


		if model_choice == 1:
			directory = 'FSPL'
			u0 = np.random.uniform(-rho,rho)
			my_own_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='FSPL')
			my_own_parameters = [t0,u0,tE,rho]


		if model_choice == 2:
			directory = 'DSPL'
			u0 = np.random.uniform(-0.5,0.5)
			u02 = np.random.uniform(-u0,u0)
			sign = np.random.choice([-1,1])
			delta_u0 = u0-sign*u02



			delta_t0 =  np.random.uniform(np.percentile(wfirst3.lightcurve_flux[:,0],20),np.percentile(wfirst3.lightcurve_flux[:,0],80))-t0
			#delta_u0 = np.random.uniform(-0.5,0.5)
			#delta_flux = np.random.uniform(0.1,1)
			delta_flux = 1
			my_own_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='DSPL')


			my_own_parameters = [t0,u0,delta_t0,delta_u0,tE,delta_flux]

		if model_choice == 3:
			directory = 'Binary'

			logs = np.random.uniform(-0.5,0.5)
			logq = np.random.uniform(-2,0)
			alpha = np.random.uniform(-np.pi,np.pi)
			my_own_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='USBL')
			my_own_model.binary_origin='central_caustic'
			location = np.random.randint(0,2)
			if location == 1:
				my_own_model.binary_origin='planetary_caustic'
			u0 = find_u0_for_caustics(10**logs,10**logq,rho,alpha,origin=my_own_model.binary_origin)

			my_own_parameters = [t0,u0,tE,rho,logs,logq,alpha]

		if model_choice == 4:
			directory = 'Planetary'

			logs = np.random.uniform(-0.5,0.5)
			logq = np.random.uniform(-6,-2)

			alpha = np.random.uniform(-np.pi,np.pi)
			my_own_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='USBL')
			my_own_model.binary_origin='central_caustic'
			location = np.random.randint(0,2)
			if location == 1:
				my_own_model.binary_origin='planetary_caustic'

			u0 = find_u0_for_caustics(10**logs,10**logq,rho,alpha,origin=my_own_model.binary_origin)

			my_own_parameters = [t0,u0,tE,rho,logs,logq,alpha]

		#my_own_parameters = microlsimulator.simulate_microlensing_model_parameters(my_own_model)

		mag_baseline = np.random.uniform(14,27)

		flux_baseline = 10**((ZP-mag_baseline)/2.5)

		g = np.random.uniform(0,1)
		f_source = flux_baseline/(1+g)



		#my_own_flux_parameters = microlsimulator.simulate_fluxes_parameters(my_own_creation.telescopes)
		my_own_flux_parameters = [f_source,g]
		my_own_parameters += my_own_flux_parameters
		#print(my_own_parameters)
		pyLIMA_parameters = my_own_model.compute_pyLIMA_parameters(my_own_parameters)

		microlsimulator.simulate_lightcurve_flux(my_own_model, pyLIMA_parameters,  red_noise_apply='No')

		lightcurve = np.c_[wfirst_tot.lightcurve_flux[:,0],ZP-2.5*np.log10(wfirst_tot.lightcurve_flux[:,1]),wfirst_tot.lightcurve_flux[:,2]/wfirst_tot.lightcurve_flux[:,1]*2.5/np.log(10)]

		if np.abs(lightcurve[:,1].max()-lightcurve[:,1].min())>1.0:
			#print(my_own_parameters)
			#plt.scatter(lightcurve[:,0],lightcurve[:,1])
			#plt.gca().invert_yaxis()
			#plt.xlim([2459700,2459820])
			#plt.show()
			params.append(my_own_parameters+['Lightcurve_'+str(count)+'.npy'])
			np.save(directory+'/Lightcurve_'+str(count)+'.npy',lightcurve)
			print(model_choice,count)
			count += 1
	np.savetxt(directory+'/LC_params.txt',np.array(params),fmt='%s')
		#if model_choice == 1:
		#	print(my_own_parameters)
			#plt.errorbar(wfirst_tot.lightcurve_flux[:,0]-2450000,ZP-2.5*np.log10(wfirst_tot.lightcurve_flux[:,1]),yerr=wfirst_tot.lightcurve_flux[:,2]/wfirst_tot.lightcurve_flux[:,1],fmt='.k')
			#plt.gca().invert_yaxis()
			#plt.show()
		#	model_fit = microlmodels.create_model('PSPL', my_own_creation)

		#	my_own_creation.fit(model_fit,'LM')
		#	my_own_creation.fits[-1].produce_outputs()
		#	plt.show()
		#plt.subplot(122)
		#trajectory_x, trajectory_y, separation = my_own_model.source_trajectory(wfirst_tot,t0, u0,pyLIMA_parameters.tE,pyLIMA_parameters)
		#plt.scatter(trajectory_x,trajectory_y,c=wfirst_tot.lightcurve_flux[:,0])
		#plt.colorbar()
		#if model_choice == 2:
		#	regime, caustics, cc = microlcaustics.find_2_lenses_caustics_and_critical_curves(
		#	    10 ** pyLIMA_parameters.logs,
		#	    10 ** pyLIMA_parameters.logq,
		#	    resolution=5000)
		#	for count, caustic in enumerate(caustics):

		#		try:
		#			plt.plot(caustic.real, caustic.imag, lw=3, c='r')
		#			plt.plot(cc[count].real, cc[count].imag, '--k')
		#		except:
		#			pass
		#plt.axis([-1,1,-1,1])
		#plt.show()
import pdb; pdb.set_trace()


print( time.time()-start)
import pdb;
pdb.set_trace()
