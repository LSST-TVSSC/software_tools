# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:38:39 2018

@author: rstreet
"""

from sys import argv
from os import path
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import LombScargle

def simulate_bd_lightcurve():
    """Function to simulate LSST DDF observations of a variable brown dwarf"""
    
    params = get_args()
    
    lci = read_luhman_lc(params['ip_lc_file'])
    lcz = read_luhman_lc(params['ip_lc_file'])
    
    vis_windows = parse_visibility_windows(params)
    
    down_lci = downsample_lightcurve(params,lci,vis_windows)
    down_lcz = downsample_lightcurve(params,lcz,vis_windows)
    
    plot_lc(lci, lcz, down_lci, down_lcz, params, 'SDSS-i', 'Pan-STARRS-Z')
    
    output_down_lc(down_lci, params, 'downsampled_lc_i.txt')
    output_down_lc(down_lcz, params, 'downsampled_lc_z.txt')
    
    run_lomb_scargle(down_lci,params,'periodogram_i.png')
    run_lomb_scargle(down_lcz,params,'periodogram_z.png')
     
def get_args():
    
    params = { }

    if len(argv) == 2:
        
        f = argv[1]
            
    else:
        print('Call sequence:')
        print('> python bd_simulator.py <path input file>')
        print('Input file format:')
        print('ip_lc_file      <file_path>')
        print('zp_lc_file      <file_path>')
        print('visibility_file <file_path>')
        print('year            <integer>')
        print('cadence         <float, mins>')
        print('output_dir      <directory_path>')
        exit()
        
    lines = open(f,'r').readlines()
    
    for l in lines:
        
        (key, value) = l.replace('\n','').split()
        
        if key in ['year']:
            params[key] = int(value)
        elif key in ['cadence']:
            params[key] = float(value)
        else:
            params[key] = value
            
    params['cadence'] = params['cadence']/(60.0*24.0) # mins -> days
    
    return params

def read_luhman_lc(file_path):
    """Function to read in the data for a BD lightcurve"""
    
    if path.isfile(file_path) == False:
        raise IOError('Cannot find '+file_path)
        exit()
    
    lines = open(file_path,'r').readlines()
    
    data = []
    
    for l in lines:
        
        if '#' not in l:
            
            entries = l.replace('\n','').split()
            
            data.append( [float(entries[1]), float(entries[2]), float(entries[3])] )
            
    data = np.array(data)
    
    return data
    
def parse_visibility_windows(params):
    """Function to read in the visibility windows, transposing the dates so 
    that they overlap the year of the observations, for the purposes of
    comparison."""
    
    if path.isfile(params['visibility_file']) == False:
        raise IOError('Cannot find '+params['visibility_file'])
        exit()
    
    lines = open(params['visibility_file'],'r').readlines()
    
    data = []

    for l in lines:
        
        if l[0:1] != '#':
            
            (date, start_time, date, end_time) = l.replace('\n','').split()
            
            start_date = Time(str(params['year'])+date[4:]+'T'+start_time, 
                                  format='isot', scale='utc')
            
            end_date = Time(str(params['year'])+date[4:]+'T'+end_time, 
                                  format='isot', scale='utc')
            
            data.append( [start_date.jd, end_date.jd] )
            
    return data

def downsample_lightcurve(params,lc,vis_windows):
    """Function to downsample a lightcurve, by selecting observations only
    within the given visibility windows, and at the cadence given"""
    
    obs_start = lc[:,0].min()
    obs_end = lc[:,0].max()
    
    tol = 30.0/(60.0*24.0)      # Tolerance on time matching, days
    
    down_lc = []
    idx_lc = []

    for (start_window, end_window) in vis_windows:

        if start_window >= obs_start and end_window <= obs_end:
            
            # Select obs data within this visible date
            idx = np.where(lc[:,0] >= start_window)[0]
            jdx = np.where(lc[:,0] <= end_window)[0]
            kdx = list(set(idx).intersection(set(jdx)))
            
            if len(kdx) > 0:
                
                ts = lc[kdx[0],0]
                
                down_lc.append( lc[kdx[0],:] )
                idx_lc.append(kdx[0])
                
                while ts <= end_window:
                    
                    ts += params['cadence']
                    
                    deltat = abs(lc[:,0] - ts)
                    
                    if deltat.min() < tol:
                        kdx2 = np.where(deltat == deltat.min())[0]
                        if kdx2[0] not in idx_lc:
                            down_lc.append(lc[kdx2[0],:])
                            idx_lc.append(kdx2[0])
            
    down_lc = np.array(down_lc)
    
    if len(down_lc) == 0:
        raise ValueError('No data left after resampling!')
        exit()
    
    return down_lc

def plot_lc(lc1, lc2,down_lc1,down_lc2,params,label1,label2):
    """Function to output the downsampled lightcurves"""
    
    yoffset = -0.1
    
    xoffset = int(down_lc1[:,0].min())
    
    fig = plt.figure(1,(10,10))
    
    plt.rc('font', size=16.0)
    plt.rc('xtick', labelsize=16.0)
    plt.rc('ytick', labelsize=16.0)
    
    plt.errorbar(down_lc1[:,0]-xoffset, down_lc1[:,1], yerr=down_lc1[:,2], 
                 marker='.', mfc='red', mec='red', ecolor='red', 
                 ms=2, mew=4, linestyle='none',label=label1)
    
    plt.plot(lc1[:,0]-xoffset, lc1[:,1], 'r-', alpha=0.2)
                 
    plt.errorbar(down_lc2[:,0]-xoffset, down_lc2[:,1]+yoffset, yerr=down_lc2[:,2], 
                 marker='.', mfc='black', mec='black', ecolor='black', 
                 ms=2, mew=4, linestyle='none',label=label2)
    
    plt.plot(lc2[:,0]-xoffset, lc2[:,1]+yoffset, 'k-', alpha=0.2)
    
    plt.xlabel('HJD - '+str(xoffset))
    plt.ylabel('Instrumental mag')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    xmin = (down_lc1[:,0]-xoffset).min()-0.5
    xmax = (down_lc1[:,0]-xoffset).max()+0.5
    plt.axis([xmin,xmax,-0.3,0.2])
    
    plt.grid()
    
    plt.legend()
    
    plt.savefig(path.join(params['output_dir'],'resampled_lc.png'), bbox_inches='tight')
    
    plt.close(1)

def output_down_lc(down_lc,params, filename):
    """Function to output a downsampled lightcurve to a file"""
    
    if path.isdir(params['output_dir']) == False:
        raise IOError('Cannot find output directory '+params['output_dir'])
        exit()
    
    f = open(path.join(params['output_dir'],filename),'w')
    
    f.write('# BJD-TBD  mag   merr\n')
    
    for i in range(0,len(down_lc),1):
        f.write(str(down_lc[i,0])+' '+str(down_lc[i,1])+' '+str(down_lc[i,2])+'\n')
    
    f.close()
    
    print('Output downsampled lightcurve to '+\
                path.join(params['output_dir'],filename))

def run_lomb_scargle(down_lc,params,filename):
    
    (freq, power) = LombScargle(down_lc[:,0], down_lc[:,1]).autopower()
    
    
    fig = plt.figure(2,(10,10))
    
    plt.rc('font', size=16.0)
    plt.rc('xtick', labelsize=16.0)
    plt.rc('ytick', labelsize=16.0)
    
    plt.plot(freq, power)
    
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    
    plt.grid()
    
    plt.savefig(path.join(params['output_dir'],filename), bbox_inches='tight')
    
    plt.close(2)
    
if __name__ == '__main__':
    
    simulate_bd_lightcurve()
    