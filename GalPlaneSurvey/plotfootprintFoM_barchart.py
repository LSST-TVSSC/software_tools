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

nvisits_tau_obs = {2.0: 7200,
            5.0: 2880,
            11.0: 1309,
            20.0: 720,
            46.5: 310,
            73.0: 197}
# Order reversed because python plots from the bottom upwards
plot_order = ['combined', 'galactic_plane', 'galactic_bulge', 'magellenic_clouds',
                'pencilbeams', 'open_clusters', 'globular_clusters', 'bonito_sfr',
                'zucker_sfr', 'clementini_stellarpops', 'xrb_priority']
plot_order.reverse()

def plot(args):

    dataset1 = parse_data_file(args['data_file1'])
    dataset2 = parse_data_file(args['data_file2'])

    compare_opsim_NObsPriority(dataset1, dataset2)
    compare_opsim_priority(dataset1, dataset2)

def compare_opsim_NObsPriority(dataset1, dataset2):

    fontsize = 40

    # The datasets are grouped by tau_obs but the nObsPriority metric
    # doesn't vary with this parameter, so we use a fiducial for
    # array selection purposes only
    tau = 11.0

    dataTable1 = dataset1[str(tau)]
    dataTable2 = dataset2[str(tau)]
    metricData1 = []
    metricData2 = []
    for map in plot_order:
        idx1 = np.where(dataTable1['maps'] == map)[0]
        idx2 = np.where(dataTable2['maps'] == map)[0]
        metricData1.append(dataTable1[idx1]['%ofNobsPriority'][0])
        metricData2.append(dataTable2[idx2]['%ofNobsPriority'][0])
    y_pos = np.arange(len(metricData1))
    bar_width = 0.5

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    plt.barh(y_pos-bar_width/3, metricData1, bar_width, color="#6C5B7B", label=dataTable1['runName'][0])
    plt.barh(y_pos+bar_width/3, metricData2, bar_width, color="#F8B195", label=dataTable2['runName'][0])
    plt.yticks(y_pos, plot_order)
    plt.subplots_adjust(left=0.35, bottom=0.08, right=0.95, top=0.95, wspace=0, hspace=0)

    plt.xlabel('% of fiducial metric value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.xlim((0.0,100.0))
    plt.grid()
    #plt.legend(fontsize=fontsize)
    plt.savefig(path.join('results',dataTable1['runName'][0]+'_'+dataTable2['runName'][0]+'_footprint_FoM_barchart.png'))

def compare_opsim_priority(dataset1, dataset2):

    fontsize = 40

    # The datasets are grouped by tau_obs but the nObsPriority metric
    # doesn't vary with this parameter, so we use a fiducial for
    # array selection purposes only
    tau_list = [5.0, 11.0]

    for tau in tau_list:
        dataTable1 = dataset1[str(tau)]
        dataTable2 = dataset2[str(tau)]
        metricData1 = []
        metricData2 = []
        overlapData1 = []
        overlapData2 = []
        for map in plot_order:
            idx1 = np.where(dataTable1['maps'] == map)[0]
            idx2 = np.where(dataTable2['maps'] == map)[0]
            metricData1.append(dataTable1[idx1]['%ofPriority'][0])
            metricData2.append(dataTable2[idx2]['%ofPriority'][0])
            overlapData1.append(dataTable1[idx2]['%overlap'][0])
            overlapData2.append(dataTable2[idx2]['%overlap'][0])
        y_pos = np.arange(len(metricData1))
        bar_width = 0.5

        fig = plt.figure(figsize=(20,20))
        ax = plt.subplot(111)
        plt.barh(y_pos-bar_width/3, metricData1, bar_width, color="#6C5B7B",
            label='% region priority, \n'+dataTable1['runName'][0])
        plt.barh(y_pos+bar_width/3, metricData2, bar_width, color="#F8B195",
            label='% region priority, \n'+dataTable2['runName'][0])
        plt.plot(overlapData1, y_pos-bar_width/3, marker='D',
                    markerfacecolor="#6C5B7B", markeredgecolor="white",
                    linestyle='None', markersize=20,
                    label='% region HEALpix')
        plt.plot(overlapData2, y_pos+bar_width/3, marker='*',
                    markerfacecolor="#F8B195", markeredgecolor="white",
                    linestyle='None', markersize=30,
                    label='% region HEALpix')
        plt.yticks(y_pos, plot_order)
        plt.subplots_adjust(left=0.35, bottom=0.08, right=0.95, top=0.95, wspace=0, hspace=0)

        plt.xlabel('%', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        #plt.xlim((0.0,100.0))
        plt.grid()
        #plt.legend(fontsize=fontsize)
        plt.savefig(path.join('results',dataTable1['runName'][0]+'_'+dataTable2['runName'][0]+'_'+str(tau)+'_footprint_priority_barchart.png'))

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    dataset = {}

    for line in file_lines[1:]:
        entries = line.replace('\n','').split()
        tau = entries[2]
        mapName = entries[1]
        runName = entries[0]
        metric1 = float(entries[10])
        metric2 = float(entries[11])
        metric3 = float(entries[5])
        if tau in dataset.keys():
            tau_data = dataset[tau]
        else:
            tau_data = {'maps': [], 'runName': [],
                        '%ofPriority': [], '%ofNobsPriority': [],
                        '%overlap': []}
        tau_data['maps'].append(mapName.replace('_map',''))
        tau_data['runName'].append(runName)
        tau_data['%ofPriority'].append(metric1)
        tau_data['%ofNobsPriority'].append(metric2)
        tau_data['%overlap'].append(metric3)
        dataset[tau] = tau_data

    for tau, tau_data in dataset.items():
        tab = Table([Column(tau_data['maps'], name='maps'),
                     Column(tau_data['runName'], name='runName'),
                     Column(tau_data['%ofPriority'], name='%ofPriority', dtype=float),
                     Column(tau_data['%ofNobsPriority'], name='%ofNobsPriority', dtype=float),
                     Column(tau_data['%overlap'], name='%overlap', dtype=float)])
        dataset[tau] = tab

    return dataset

if __name__ == '__main__':
    args = {}
    if len(argv) == 1:
        args['data_file1'] = input('Please enter the path to the first data file: ')
        args['data_file2'] = input('Please enter the path to the second data file: ')
    else:
        args['data_file1'] = argv[1]
        args['data_file2'] = argv[2]

    plot(args)
