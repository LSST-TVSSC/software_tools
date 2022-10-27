from os import path
from sys import argv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap
from astropy.table import Table, Column


def plot(args):

    dataset = parse_data_file(args['data_file'])

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.35, bottom=0.4, right=1.3, top=0.8)

    #fig.patch.set_facecolor("white")

    plot_colours = {'u': '#0003FE', 'g': '#00C939',
                    'r': '#FCA400', 'i': '#FF6C00',
                    'z': '#950000', 'y': '#B100B4'}
    plot_symbols = {'u': 'o', 'g': 'v',
                    'r': 'P', 'i': 's',
                    'z': 'D', 'y': 'X'}
    filter_list = ['u','g','r','i','z','y']
    # Order reversed because python plots from the bottom upwards
    plot_order = ['combined', 'galactic_plane', 'galactic_bulge', 'magellenic_clouds',
                    'pencilbeams', 'open_clusters', 'globular_clusters', 'bonito_sfr',
                    'zucker_sfr', 'clementini_stellarpops', 'xrb_priority']
    plot_order.reverse()

    y_pos = np.arange(0,len(plot_order),1)
    barwidth = 0.05
    baroffset = 0.1
    fontsize = 40
    y_pos_offsets = np.array([((baroffset*len(filter_list))/2.0)]*len(plot_order))

    # Loop over each science map:
    for m,mapName in enumerate(plot_order):

        # ...and filter, to ensure the correct colours and symbols are used in
        # each sequence.
        for b,bandpass in enumerate(filter_list):
            # Extract the relevant data for each map from the dataset:
            map_data = getMapData(dataset, args, mapName, bandpass)

            # Plot a background bar line:
            plt.barh(y_pos[m]+(b*baroffset), map_data['metric'][0], barwidth,
                     color=plot_colours[bandpass])

            # Plot the data
            if m == 0:
                plt.plot(map_data['metric'], [y_pos[m]+(b*baroffset)], marker=plot_symbols[bandpass],
                markerfacecolor=plot_colours[bandpass], markeredgecolor=plot_colours[bandpass],
                linestyle='None', markersize=20, alpha=0.7, label=bandpass)
            else:
                plt.plot(map_data['metric'], [y_pos[m]+(b*baroffset)], marker=plot_symbols[bandpass],
                markerfacecolor=plot_colours[bandpass], markeredgecolor=plot_colours[bandpass],
                linestyle='None', markersize=20, alpha=0.7)

    # Label the bars
    plt.yticks(y_pos+y_pos_offsets, plot_order, fontsize=fontsize, color='grey')
    plt.xlabel('% of region with $R_{exp}\geq$'+str(round(args['threshold']/100.0,2)), fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlim((0.0,100.0))
    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([xmin,xmax,ymin,ymax*1.05])
    plt.grid()
    plt.legend(loc='upper right', ncol=len(filter_list), fontsize=fontsize*0.7)

    # Plot legend outside the plot face area
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * -0.001,
    #         box.width, box.height * 0.95])

    #loc = (0.25, 1.05)
    #l = ax.legend(loc='upper center', bbox_to_anchor=loc,
    #                ncol=len(filter_list), fontsize=fontsize*0.9)

    #l.legendHandles[0]._sizes = [50]
    #if len(l.legendHandles) > 1:
    #    l.legendHandles[1]._sizes = [50]

    plt.tight_layout()
    plot_file = path.join(args['output_dir'],dataset['runName'][0]+'_timeperfilter_barchart.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close(1)

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    runName = []
    mapName = []
    bandpass = []
    mean_ratio = []
    min_ratio = []
    max_ratio = []
    median_ratio = []
    stddev_ratio = []
    npix_50 = []
    percent_50 = []
    npix_75 = []
    percent_75 = []
    npix_90 = []
    percent_90 = []
    npix_100 = []
    percent_100 = []
    sum_ratio = []

    for line in file_lines[1:]:
        entries = line.replace('\n','').split()
        runName.append(entries[0])
        mapName.append(entries[1])
        bandpass.append(entries[2])
        mean_ratio.append(entries[3])
        min_ratio.append(entries[4])
        max_ratio.append(entries[5])
        median_ratio.append(entries[6])
        stddev_ratio.append(entries[7])
        npix_50.append(entries[8])
        percent_50.append(entries[9])
        npix_75.append(entries[10])
        percent_75.append(entries[11])
        npix_90.append(entries[12])
        percent_90.append(entries[13])
        npix_100.append(entries[14])
        percent_100.append(entries[15])
        sum_ratio.append(entries[16])

    dataset = Table([Column(runName, name='runName'),
                 Column(mapName, name='maps'),
                 Column(bandpass, name='bandpass'),
                 Column(mean_ratio, name='mean_ratio', dtype=float),
                 Column(mean_ratio, name='min_ratio', dtype=float),
                 Column(mean_ratio, name='max_ratio', dtype=float),
                 Column(mean_ratio, name='median_ratio', dtype=float),
                 Column(stddev_ratio, name='stddev_ratio', dtype=float),
                 Column(npix_50, name='npix_50', dtype=float),
                 Column(percent_50, name='percent_50', dtype=float),
                 Column(npix_75, name='npix_75', dtype=float),
                 Column(percent_75, name='percent_75', dtype=float),
                 Column(npix_90, name='npix_90', dtype=float),
                 Column(percent_90, name='percent_90', dtype=float),
                 Column(npix_90, name='npix_100', dtype=float),
                 Column(percent_100, name='percent_100', dtype=float),
                 Column(sum_ratio, name='sum_ratio', dtype=float)])

    print(dataset)
    return dataset


def getMapData(dataset, params, map, bandpass):

    # Fet the metric values for each science map for this run and filter:
    colname = 'percent_'+str(int(params['threshold']))
    #colname = 'sum_ratio'
    data = []
    datum = dataset[colname][(dataset['maps'] == map+'_map') & (dataset['bandpass'] == bandpass)]
    data.append(datum.data)

    return Table([Column([map], name='map'),
                  Column(data, name='metric', dtype=float)])

if __name__ == '__main__':

    args = {}
    if len(argv) == 1:
        args['data_file'] = input('Please enter the path to the data file: ')
        args['threshold'] = float(input('Please enter the threshold to plot {50, 75, 90}: '))
        args['output_dir'] = input('Please enter the path to the output directory: ')
    else:
        args['data_file'] = argv[1]
        args['threshold'] = float(argv[2])
        args['output_dir'] = argv[3]

    plot(args)
