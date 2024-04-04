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
science_maps = ['combined', 'galactic_plane', 'galactic_bulge', 'magellenic_clouds',
                'pencilbeams', 'open_clusters', 'globular_clusters', 'bonito_sfr',
                'zucker_sfr', 'clementini_stellarpops', 'xrb_priority']
#plot_order.reverse()

tau_symbols = {2.0: 'o', 5.0: 'v', 11.0: 'P', 20.0: 's', 46.5: 'D', 73.0: 'X'}
tau_colours1 = {2.0: '#00282E', 5.0: '#004851', 11.0: '#006C7A', 20.0: '#26C6DA', 46.5: '#00BED6', 73.0: '#6BEFF9'}
tau_colours2 = {2.0: '#5D2818', 5.0: '#853A22', 11.0: '#C25432', 20.0: '#FF7043', 46.5: '#FF9776', 73.0: '#FFBEA9'}
tau_colours3 = {2.0: '#9f00fe', 5.0: '#b147fb', 11.0: '#c06cf8', 20.0: '#cc8df5', 46.5: '#d7acf1', 73.0: '#e0caec'}

def plot(args):

    sim_list = load_sims_list(args)
    plot_order = build_plot_order(sim_list)

    sim_data = {}
    for sim in sim_list:
        sim_name = path.basename(sim).replace('_survey_cadence_data.txt','')
        dataset = parse_data_file(sim, args['science_map'])
        sim_data[sim_name]= dataset

    compare_cadence_FoMpercent(plot_order, sim_data, 'VIM', args)
    compare_cadence_FoMpercent(plot_order, sim_data, 'SVGM', args)

def compare_cadence_FoMpercent(plot_order, sim_data, metricName, args):

    fontsize = 40
    y_pos = np.arange(0,len(plot_order),1)
    barwidth = 0.05
    baroffset = 0.2

    fig = plt.figure(figsize=(20,len(plot_order)+2))
    ax = plt.subplot(111)
    k = 0
    for sim_name in plot_order:
        dataTable = sim_data[sim_name]
        print(sim_name, dataTable,metricName.lower())

        # Plots horizontal guideline
        plt.barh(y_pos[k], 100.0, barwidth, color="#D5D1D1")

        if metricName == 'VIM':
            for tau in nvisits_tau_obs.keys():
                idx = np.where(dataTable['tau'] == tau)
                if k == 0:
                    label = '$\\tau_{obs}$='+str(tau)
                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[k]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20, label=label)
                else:
                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[k]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20)

        else:
            for tau in [20.0, 46.5, 73.0]:
                idx = np.where(dataTable['tau'] == tau)
                if k == 0:
                    label = '$\\tau_{var}$='+str(dataTable['tau_var'][idx][0])

                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[k]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20, label=label)
                else:
                    plt.plot(dataTable[metricName.lower()][idx], [y_pos[k]], marker=tau_symbols[tau],
                    markerfacecolor=tau_colours1[tau], markeredgecolor=tau_colours1[tau],
                    linestyle='None', markersize=20)
        k += 1

    plt.yticks(y_pos, plot_order)
    plt.subplots_adjust(left=0.225, bottom=0.2, right=0.95, top=0.80, wspace=0, hspace=0)

    plt.title(metricName, fontsize=fontsize)
    plt.xlabel('% of ideal', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=30)
    plt.xlim((0.0,100.0))
    plt.grid()
    #plt.legend(fontsize=fontsize)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.001,
             box.width, box.height * 0.95])

    if metricName == 'VIM':
        loc = (0.35, 1.4)
    else:
        loc = (0.25, 1.12)
    l = ax.legend(loc='upper center', bbox_to_anchor=loc,
                    ncol=len(nvisits_tau_obs), fontsize=fontsize*0.6)

    l.legendHandles[0]._sizes = [50]
    if len(l.legendHandles) > 1:
        l.legendHandles[1]._sizes = [50]

    plt.savefig(path.join(args['output_dir'],args['plot_name']+'_'+args['science_map']+'_'+metricName+'_barchart.png'))

def parse_data_file(data_file, science_map):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    mapMetrics = {'runName': [], 'tau': [], 'tau_var': [],
                'vim': [], 'svgm': [],
                'vip': []}

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
            if science_map in mapName:
                mapMetrics['tau'].append(tau)
                mapMetrics['tau_var'].append(tau_var)
                mapMetrics['runName'].append(runName)
                mapMetrics['vim'].append(vim)
                mapMetrics['svgm'].append(svgm)
                mapMetrics['vip'].append(vip)


    dataset = Table([Column(mapMetrics['tau'], name='tau'),
                 Column(mapMetrics['tau_var'], name='tau_var'),
                 Column(mapMetrics['runName'], name='runName'),
                 Column(mapMetrics['vim'], name='vim', dtype=float),
                 Column(mapMetrics['svgm'], name='svgm', dtype=float),
                 Column(mapMetrics['vip'], name='vip', dtype=float)])

    return dataset


def parse_data_file_per_tau(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    mapMetrics = {'runName': [], 'tau': [], 'tau_var': [],
                'vim': [], 'svgm': [],
                'vip': []}
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

            if str(tau) in dataset.keys():
                tau_data = dataset[str(tau)]
            else:
                tau_data = {'maps': [], 'runName': [], 'tau': [], 'tau_var': [],
                            'vim': [], 'svgm': [], 'vip': []}

            tau_data['maps'].append(mapName)
            tau_data['runName'].append(runName)
            tau_data['tau'].append(tau)
            tau_data['tau_var'].append(tau_var)
            tau_data['vim'].append(vim)
            tau_data['svgm'].append(svgm)
            tau_data['vip'].append(vip)
            dataset[str(tau)] = tau_data

    for tau, tau_data in dataset.items():
        tab = Table([Column(tau_data['maps'], name='maps'),
                     Column(tau_data['runName'], name='runName'),
                     Column(tau_data['tau'], name='tau'),
                     Column(tau_data['tau_var'], name='tau_var'),
                     Column(tau_data['vim'], name='vim', dtype=float),
                     Column(tau_data['svgm'], name='svgm', dtype=float),
                     Column(tau_data['vip'], name='vip', dtype=float)])
        dataset[tau] = tab

    return dataset

def load_sims_list(args):
    if not path.isfile(args['sim_list']):
        raise IOError('Cannot find input list of opsims: '+args['sim_list'])

    file_lines = open(args['sim_list'],'r').readlines()
    sim_list = []
    for line in file_lines:
        entries = line.replace('\n','').split()
        sim_list.append(entries[1])

    return sim_list

def build_plot_order(sim_list):
    plot_order = []
    for sim in sim_list:
        plot_order.append(path.basename(sim.replace('_survey_cadence_data.txt','')))
    plot_order.reverse()
    return plot_order

if __name__ == '__main__':
    args = {}
    if len(argv) == 1:
        args['sim_list'] = input('Please enter the path to the list of opsims: ')
        args['plot_name'] = input('Please enter the name for the output plot: ')
        print('List of science maps available: \n' + repr(science_maps))
        args['science_map'] = input('Please select the science map to plot for: ')
        args['output_dir'] = input('Please enter the path to the directory for output: ')
    else:
        args['sim_list'] = argv[1]
        args['plot_name'] = argv[2]
        args['science_map'] = argv[3]
        args['output_dir'] = argv[4]

    plot(args)
