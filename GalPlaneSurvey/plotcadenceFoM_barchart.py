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

tau_symbols = {2.0: 'o', 5.0: 'v', 11.0: 'P', 20.0: 's', 46.5: 'D', 73.0: 'X'}
tau_colours1 = {2.0: '#00282E', 5.0: '#004851', 11.0: '#006C7A', 20.0: '#26C6DA', 46.5: '#00BED6', 73.0: '#6BEFF9'}
tau_colours2 = {2.0: '#5D2818', 5.0: '#853A22', 11.0: '#C25432', 20.0: '#FF7043', 46.5: '#FF9776', 73.0: '#FFBEA9'}
tau_colours3 = {2.0: '#9f00fe', 5.0: '#b147fb', 11.0: '#c06cf8', 20.0: '#cc8df5', 46.5: '#d7acf1', 73.0: '#e0caec'}

def plot(args):

    dataset1 = parse_data_file(args['data_file1'])
    #dataset2 = parse_data_file(args['data_file2'])

    compare_cadence_FoMpercent(dataset1,'VIM')
    compare_cadence_FoMpercent(dataset1,'SVGM')

def compare_cadence_FoMpercent(dataset,metricName):

    fontsize = 40
    y_pos = np.arange(0,len(plot_order),1)
    barwidth = 0.05
    baroffset = 0.2

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    for m,mapName in enumerate(plot_order):
        dataTable = dataset[mapName]
        print(mapName, dataTable,metricName.lower())

        # Plot VIM % of ideal
        plt.barh(y_pos[m], 100.0, barwidth, color="#D5D1D1")

        if metricName == 'VIM':
            for tau in nvisits_tau_obs.keys():
                idx = np.where(dataTable['tau'] == tau)
                if m == 0:
                    label = '$\\tau_{obs}$='+str(tau)
                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[m]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20, label=label)
                else:
                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[m]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20)

        else:
            for tau in [20.0, 46.5, 73.0]:
                idx = np.where(dataTable['tau'] == tau)
                if m == 0:
                    label = '$\\tau_{var}$='+str(dataTable['tau_var'][idx][0])

                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[m]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20, label=label)
                else:
                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[m]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20)

    plt.yticks(y_pos, plot_order)
    plt.subplots_adjust(left=0.35, bottom=0.08, right=0.95, top=0.95, wspace=0, hspace=0)

    plt.title(metricName, fontsize=fontsize)
    plt.xlabel('% of ideal', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim((0.0,100.0))
    plt.grid()
    #plt.legend(fontsize=fontsize)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.001,
             box.width, box.height * 0.95])

    if metricName == 'VIM':
        loc = (0.3, 1.1)
    else:
        loc = (0.25, 1.1)
    l = ax.legend(loc='upper center', bbox_to_anchor=loc,
                    ncol=len(nvisits_tau_obs), fontsize=fontsize*0.6)

    l.legendHandles[0]._sizes = [50]
    if len(l.legendHandles) > 1:
        l.legendHandles[1]._sizes = [50]

    plt.savefig(path.join('cadence_results',dataTable['runName'][0]+'_cadence_'+metricName+'_barchart.png'))

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    dataset = {}

    for line in file_lines[1:]:
        if line[0:1] != '#':
            entries = line.replace('\n','').split()
            tau = float(entries[2])
            tau_var = float(entries[3])
            mapName = entries[1].replace('_map','')
            runName = entries[0]
            vim = float(entries[7])
            svgm = float(entries[11])
            vip = float(entries[12])
            if mapName in dataset.keys():
                mapMetrics = dataset[mapName]
            else:
                mapMetrics = {'runName': [], 'tau': [], 'tau_var': [],
                            'vim': [], 'svgm': [],
                            'vip': []}
            mapMetrics['tau'].append(tau)
            mapMetrics['tau_var'].append(tau_var)
            mapMetrics['runName'].append(runName)
            mapMetrics['vim'].append(vim)
            mapMetrics['svgm'].append(svgm)
            mapMetrics['vip'].append(vip)
            dataset[mapName] = mapMetrics

    for mapName, mapMetrics in dataset.items():
        tab = Table([Column(mapMetrics['tau'], name='tau'),
                     Column(mapMetrics['tau_var'], name='tau_var'),
                     Column(mapMetrics['runName'], name='runName'),
                     Column(mapMetrics['vim'], name='vim', dtype=float),
                     Column(mapMetrics['svgm'], name='svgm', dtype=float),
                     Column(mapMetrics['vip'], name='vip', dtype=float)])
        dataset[mapName] = tab

    return dataset

if __name__ == '__main__':
    args = {}
    if len(argv) == 1:
        args['data_file1'] = input('Please enter the path to the first data file: ')
        #args['data_file2'] = input('Please enter the path to the second data file: ')
    else:
        args['data_file1'] = argv[1]
        #args['data_file2'] = argv[2]

    plot(args)
