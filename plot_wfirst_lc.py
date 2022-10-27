# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 12:22:01 2018

@author: rstreet
"""

from os import path
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def plot_lc():
    
    data_file = get_args()
    
    data = read_lc_file(data_file)
    
    output_plot(data)
    
def get_args():
    
    if len(argv) == 1:
        
        data_file = raw_input('Please enter the path to the data file: ')
        
    else:
        
        data_file = argv[1]
    
    return data_file

def read_lc_file(data_file):
    
    if path.isfile(data_file) == False:
        
        print('ERROR: Cannot find input file '+data_file)
        
        exit()
        
    
    file_lines = open(data_file,'r').readlines()
    
    data = []
    
    for line in file_lines:
        
        entries = line.replace('\n','').split()
        
        row = []
        
        for item in entries:
            
            row.append(float(item))
        
        data.append(row)
    
    return np.array(data)

def output_plot(data):
    
    fig = plt.figure(1,(10,10))
    
    plt.rc('font', size=18.0)
    plt.rc('xtick', labelsize=18.0)
    plt.rc('ytick', labelsize=18.0)

    dt = int(data[0,0])
    
    plt.errorbar(data[:,0]-dt,data[:,1], yerr=data[:,2], fmt='.', )
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])
    
    plt.grid()
    
    plt.xlabel('HJD - '+str(dt)+' [days]')
    plt.ylabel('Mag')
    
    plt.savefig('wfirst_lc_plot.png', bbox_inches='tight')
    

if __name__ == '__main__':
    
    plot_lc()
    