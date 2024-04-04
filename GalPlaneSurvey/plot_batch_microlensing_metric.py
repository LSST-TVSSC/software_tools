from os import path
import argparse
from astropy.table import Table, Column
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

# Configuration
EVENT_TIMESCALES = [30.0, 200.0]
MICROLENSING_METRICS = ['numberTotalEventsDetected',
                        'numberTotalEvents',
                        'percentTotalEvents',
                        'medianPercentEventsHEALpixel',
                        'stddevPercentEventsHEALpixel']
def plot_microlensing_results(args):
    """
    Function to plot a heatmap comparing the results of microlensing metrics for different
    desired survey regions across different opsims
    """

    # The pre-calculate metric results for each opsim are stored in separate files.
    # Read in the list of files
    opsim_results = parse_sim_results_list(args)

    # Load the metric data for each opsim and survey region, then
    # select the data for the specific metric in a format
    # convenient for plotting
    results_data, map_list = parse_opsim_results(opsim_results)
    metric_data = select_metric_data(args, results_data)

    # Plot a heatmap of the metric results for the requested metric and event timescale
    plot_heat_map(args, metric_data, map_list)

def plot_heat_map(args, metric_data, map_list):

    opsim_list = metric_data.keys()

    # Decide whether to log zscale:
    if 'number' in args.metric:
        logdata = True
    else:
        logdata = False

    # Note the data array is maps x opsims because the orientation
    # of the plotted image will have the opsims on the x-axis
    # The x-axis starts from the origin of 0 then increases towards the right
    # The y-axis starts from an origin of max then increases towards the upward
    data_grid = np.zeros( (len(map_list),len(opsim_list)) )

    for isim,opsim in enumerate(opsim_list):
        dataset = metric_data[opsim]
        for imap,map in enumerate(map_list):
            idx = np.where(dataset['maps'] == map)[0][0]
            if logdata:
                data_grid[imap,isim] = np.log10(dataset[idx][args.metric])
            else:
                data_grid[imap, isim] = dataset[idx][args.metric]
            #print(opsim, map, dataset[idx][args.metric])

    # The first + 1 increases the length
    xgrid = np.arange(0,len(opsim_list)+1,1)
    ygrid = np.arange(0,len(map_list)+1,1)

    norm = mpl.colors.Normalize(data_grid.min(), data_grid.max())

    fig, ax = plt.subplots(figsize=(len(opsim_list)+1,len(map_list)+1))
    ax.pcolormesh(xgrid, ygrid, data_grid, cmap="magma", norm=norm)
    ax.set_frame_on(False) # remove all spines

    ax.set_xticks(xgrid[0:-1]+0.5)
    ax.set_yticks(ygrid[0:-1]+0.5)
    ax.set_ylabel('Survey region', fontsize=20)
    ax.set_title('Microlensing event timescales=' + args.tau + 'd')
    ax.set_xticklabels(opsim_list,rotation=45.0,horizontalalignment='right',fontsize=20)
    rev_tau = map_list
    #rev_science_maps.reverse()
    ax.set_yticklabels(rev_tau,fontsize=20,horizontalalignment='right')
    # Remove ticks by setting their length to 0
    #ax.yaxis.set_tick_params(length=0)
    #ax.xaxis.set_tick_params(length=0)

    # Add a colorbar scale, coords=(x coordinate of left border,
    #  y coordinate for bottom border,
    #  width,
    #  height)
    fig.subplots_adjust(bottom=0.25, left=0.1, right=0.8)
    #cbar_ax = fig.add_axes([0.3, -0.05, 0.4, 0.025])

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="magma"),
        ax=ax, # Pass the new axis
        orientation = "vertical")

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0)
    cb.ax.tick_params(labelsize=20)

    # Set legend label
    if logdata:
        cb.set_label('log10('+args.metric+')', size=10)
    else:
        cb.set_label(args.metric, size=10)

    plt.tight_layout()
    plot_file = path.join(args.output_dir,
                          args.metric+'_'+args.tau+'_opsim_microlensing_comparison_heatmap.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close()

def select_metric_data(args, results_data):

    metric_data = {}

    for opsim, dataset in results_data.items():
        tau_data = dataset[args.tau]
        metric_data[opsim] = Table([tau_data['maps'],
                                    tau_data[args.metric]])

    return metric_data
def parse_opsim_results(opsim_results):
    """
    Function to load the metric data for a set of results files for multiple opsims
    """

    results_data = {}
    map_list = []
    for opsim, results_file in opsim_results.items():
        dataset = parse_microlensing_results(results_file)
        results_data[opsim] = dataset

        # Extract a list of the maps included in the results.
        # This is expected to be the same for all opsims run.
        if len(map_list) == 0:
            tau_data = dataset[str(EVENT_TIMESCALES[0])]
            map_list = tau_data['maps'].data

    return results_data, map_list

def parse_microlensing_results(data_file):
    """
    Function to load the results of the microlensing metric calculated per survey regions,
    for a single opsim.

    Expected file format is:
    # Col 1: runName
    # Col 2: mapName
    # Col 3: tau
    # Col 4: numberTotalEventsDetected
    # Col 5: numberTotalEvents
    # Col 6: percentTotalEvents
    # Col 7: median percent events detected per HEALpixel
    # Col 8: stddev percent events detected per HEALpixel
    """

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    dataset = {}

    for line in file_lines:
        if '#' not in line[0:2]:
            entries = line.replace('\n','').split()
            tau = entries[2]
            mapName = entries[1]
            runName = entries[0]
            metric1 = float(entries[3])
            metric2 = float(entries[4])
            metric3 = float(entries[5])
            metric4 = float(entries[6])
            metric5 = float(entries[7])
            if tau in dataset.keys():
                tau_data = dataset[tau]
            else:
                tau_data = {'maps': [], 'runName': [],
                            'numberTotalEventsDetected': [], 'numberTotalEvents': [],
                            'percentTotalEvents': [], 'medianPercentEventsHEALpixel': [],
                            'stddevPercentEventsHEALpixel': []}
            tau_data['maps'].append(mapName.replace('_map',''))
            tau_data['runName'].append(runName)
            tau_data['numberTotalEventsDetected'].append(metric1)
            tau_data['numberTotalEvents'].append(metric2)
            tau_data['percentTotalEvents'].append(metric3)
            tau_data['medianPercentEventsHEALpixel'].append(metric4)
            tau_data['stddevPercentEventsHEALpixel'].append(metric5)
            dataset[tau] = tau_data

    for tau, tau_data in dataset.items():
        tab = Table([Column(tau_data['maps'], name='maps'),
                     Column(tau_data['runName'], name='runName'),
                     Column(tau_data['numberTotalEventsDetected'], name='numberTotalEventsDetected', dtype=float),
                     Column(tau_data['numberTotalEvents'], name='numberTotalEvents', dtype=float),
                     Column(tau_data['percentTotalEvents'], name='percentTotalEvents', dtype=float),
                     Column(tau_data['medianPercentEventsHEALpixel'], name='medianPercentEventsHEALpixel', dtype=float),
                     Column(tau_data['stddevPercentEventsHEALpixel'], name='stddevPercentEventsHEALpixel', dtype=float),])
        dataset[tau] = tab

    return dataset

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('sim_results_list', help='List of files containing microlensing metric results for different OpSims')
    parser.add_argument('metric', help='Name of the metric to plot, one of: ' + repr(MICROLENSING_METRICS))
    parser.add_argument('tau', help='Event timescale to be plotted, one of: ' + repr(EVENT_TIMESCALES))
    parser.add_argument('output_dir', help='Path to output directory')
    args = parser.parse_args()

    if float(args.tau) not in EVENT_TIMESCALES:
        raise IOError('Requested event timescale (' + args.tau + ') is not in allowed set (' + repr(EVENT_TIMESCALES) + ')')

    if args.metric not in MICROLENSING_METRICS:
        raise IOError('Requested metric (' + args.metric + ') is not one of those available (' + repr(MICROLENSING_METRICS) + ')')

    return args

def parse_sim_results_list(args):
    """The list of opsim results is a text file consisting of a list of entries
    of the form:
    opsim_identifier        path_to_results_file
    """

    if not path.isfile(args.sim_results_list):
        raise IOError('Cannot find the list of OpSim results files at ' + args.sim_results_list)

    data_list = open(args.sim_results_list,'r').readlines()

    opsim_results = {}
    for line in data_list:
        entries = line.replace('\n','').split()
        opsim_results[entries[0]] = entries[1]

    return opsim_results

if __name__ == '__main__':
    args = get_args()
    plot_microlensing_results(args)