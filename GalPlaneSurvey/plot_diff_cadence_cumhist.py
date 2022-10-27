from os import path
from sys import argv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap
from astropy.table import Table, Column

tau_obs = [2.0, 5.0, 11.0, 20.0, 46.5, 73.0]
tau_colours3 = {2.0: '#26C6DA', 5.0: '#112E51', 11.0: '#FF7043',
                20.0: '#78909C', 46.5: '#2E78D2', 73.0: '#FFBEA9'}
output_dir = './cadence_results'


def plot(args):

    dataset1 = parse_data_file(args['data_file1'])
    runName1 = '_'.join(path.basename(args['data_file1']).split('_')[0:3])
    dataset2 = parse_data_file(args['data_file2'])
    runName2 = '_'.join(path.basename(args['data_file2']).split('_')[0:3])

    deltas = calc_deltas(dataset1,dataset2)

    fig = plt.figure(1,figsize=(10,10))
    plt.rcParams['font.size'] = 22
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

    xdata = deltas[:,0]
    for i,tau in enumerate(tau_obs):
        plt.plot(xdata, deltas[:,i+1], color=tau_colours3[tau], ls='-', label='$\\tau_{obs}$='+str(tau))
    #plt.plot(xdata, deltas[:,-1], color='k', ls='-.', label='Ideal')

    #(xmin,xmax,ymin,ymax) = plt.axis()
    #xmax += 2.5
    #plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel('HEALpixel priority')
    plt.ylabel('Delta cumulative VIM')
    plt.grid()
    #plt.legend()
    plt.tight_layout()
    plt.savefig(path.join(output_dir, runName1+'_'+runName2+'_diffcumulativeVIM.png'))
    plt.close(1)

def parse_data_file(file_path):

    if not path.isfile(file_path):
        raise IOError('Cannot find file '+file_path)

    file_lines = open(file_path,'r').readlines()

    data = []
    for line in file_lines:
        items = line.replace('\n','').split()
        entry = []
        for datum in items:
            entry.append(float(datum))
        data.append(entry)
    data = np.array(data)

    #column_list = [Column(data[:,0], name='pixpriority', dtype=float)]
    #for i,tau in enumerate(tau_obs):
    #    column_list.append( Column(data[:,i+1], name=str(tau), dtype=float) )
    #column_list.append( Column(data[:,-1], name='ideal', dtype=float) )

    return data

def calc_deltas(dataset1,dataset2):

    deltas = np.zeros(dataset1.shape)
    deltas[:,0] = dataset1[:,0]
    deltas[:,-1] = dataset1[:,-1]
    for i,tau in enumerate(tau_obs):
        deltas[:,i+1] = dataset1[:,i+1] - dataset2[:,i+1]

    return deltas

if __name__ == '__main__':
    args = {}
    if len(argv) == 1:
        args['data_file1'] = input('Please enter the path to the first data file: ')
        args['data_file2'] = input('Please enter the path to the second data file: ')
    else:
        args['data_file1'] = argv[1]
        args['data_file2'] = argv[2]

    plot(args)
