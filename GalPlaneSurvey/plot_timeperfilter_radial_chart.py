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

    (fig, ax) = plt.subplots(1, 1, figsize=(10, 9.5),
                                  subplot_kw={"projection": "polar"})
    plt.subplots_adjust(left=0.3, right=0.9)

    fig.patch.set_facecolor("white")

    ymax = 100.0
    ax.set_facecolor("white")
    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(0, ymax)

    plot_colours = {'u': '#0003FE', 'g': '#00C939',
                    'r': '#FCA400', 'i': '#FF6C00',
                    'z': '#950000', 'y': '#B100B4'}
    plot_symbols = {'u': 'o', 'g': 'v',
                    'r': 'P', 'i': 's',
                    'z': 'D', 'y': 'X'}
    filter_list = ['u','g','r','i','z','y']

    # Loop over each bandpass since radial plots are generated for all
    # science maps:
    for bandpass in filter_list:

        # Extract the relevant data for each map from the dataset:
        map_data = getMapData(dataset, args, bandpass)

        # Values for the x axis
        theta = np.linspace(0.05, 2 * np.pi - 0.05, len(map_data), endpoint=False)

        # Plot the data
        ax.plot(theta, map_data['metric'], color=plot_colours[bandpass], linestyle='-', alpha=0.5, zorder=10)
        ax.scatter(theta, map_data['metric'], color=plot_colours[bandpass],
                    marker=plot_symbols[bandpass], s=120, zorder=10, label=bandpass)

    # Label the bars
    labels = ["\n".join(wrap(c, 10, break_long_words=False)) for c in map_data['maps']]
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, size=18, color='grey');
    #ax.set_title('NVisits = '+str(nvisits)+', $\\tau_{obs}$='+str(tau_obs[nvisits])+'days')

    # Plot radial gridlines
    ax.vlines(theta, 0, ymax, color="#1f1f1f", ls=(0, (4, 4)), zorder=11)

    # Plot legend outside the plot face area
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.001,
             box.width, box.height * 0.95])

    loc = (0.45, 1.15)
    l = ax.legend(loc='upper center', bbox_to_anchor=loc,
                    ncol=len(filter_list), fontsize=16)

    l.legendHandles[0]._sizes = [50]
    if len(l.legendHandles) > 1:
        l.legendHandles[1]._sizes = [50]

    plt.tight_layout()
    plot_file = path.join(args['output_dir'],dataset['runName'][0]+'_timeperfilter_radial_plot.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close()

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    runName = []
    mapName = []
    bandpass = []
    mean_ratio = []
    stddev_ratio = []
    npix_50 = []
    percent_50 = []
    npix_75 = []
    percent_75 = []
    npix_90 = []
    percent_90 = []

    for line in file_lines[1:]:
        entries = line.replace('\n','').split()
        runName.append(entries[0])
        mapName.append(entries[1])
        bandpass.append(entries[2])
        mean_ratio.append(entries[3])
        stddev_ratio.append(entries[4])
        npix_50.append(entries[5])
        percent_50.append(entries[6])
        npix_75.append(entries[7])
        percent_75.append(entries[8])
        npix_90.append(entries[9])
        percent_90.append(entries[10])

    dataset = Table([Column(runName, name='runName'),
                 Column(mapName, name='maps'),
                 Column(bandpass, name='bandpass'),
                 Column(mean_ratio, name='mean_ratio', dtype=float),
                 Column(stddev_ratio, name='stddev_ratio', dtype=float),
                 Column(npix_50, name='npix_50', dtype=float),
                 Column(percent_50, name='percent_50', dtype=float),
                 Column(npix_75, name='npix_75', dtype=float),
                 Column(percent_75, name='percent_75', dtype=float),
                 Column(npix_90, name='npix_90', dtype=float),
                 Column(percent_90, name='percent_90', dtype=float)])

    print(dataset)
    return dataset


def getMapData(dataset, params, bandpass):

    # Get a list of the science maps in the dataset
    scienceMaps = []
    for map in dataset['maps']:
        if map not in scienceMaps:
            scienceMaps.append(map)

    # Fet the metric values for each science map for this run and filter:
    colname = 'percent_'+str(int(params['threshold']))
    data = []
    for map in scienceMaps:
        datum = dataset[colname][(dataset['maps'] == map) & (dataset['bandpass'] == bandpass)]
        data.append(datum.data)

    return Table([Column(scienceMaps, name='maps'),
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
