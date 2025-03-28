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
import plotfootprintFoM_barchart
import plotcadenceFoM_compare_opsims
import plot_timeperfilter_barchart
import plot_period_color_metrics
import plot_annual_cadence

SCIENCE_MAPS = ['combined', 'galactic_plane', 'galactic_bulge', 'magellenic_clouds',
                'pencilbeams', 'open_clusters', 'globular_clusters', 'bonito_sfr',
                'zucker_sfr', 'clementini_stellarpops', 'xrb_priority']
FOOTPRINT_METRICS = ['%ofPriority', '%ofNobsPriority','%overlap']
CADENCE_METRICS = ['vim', 'svgm', 'vip']
FILTER_METRICS = ['percent_100']
PERIOD_COLOR_METRICS = ['%periodDetect', 'medianUniformity', 'medianNYSOs', 'medianfTgaps']
ANNUAL_CADENCE_METRICS = ['vim_year1','vim_year2','vim_year3','vim_year4',\
                          'vim_year5','vim_year6','vim_year7','vim_year8',\
                          'vim_year9','vim_year10']
TAU_OBS = [2.0, 5.0, 11.0, 20.0, 46.5, 73.0]

def plot_metric_radial_charts(params):

    # Parse the list of simulation results
    params = parse_sim_results_list(params)

    # Parse the datafiles, extracting the data on the requested metric, and
    # applying the requested constraint
    results_data = parse_results_files(params)
    metric_data = select_metric_data(params, results_data)

    # Generate a radial plot of the results
    #plot_radial_chart(params,metric_data)
    plot_heat_map(params,metric_data,zmax=100.0)

def get_args():

    params = {}
    if len(argv) == 1:
        params['sim_results_list'] = input('Please enter the path to the list of OpSim results files: ')
        params['metric'] = input('Please enter the identifier of the metric (column name) to extract from the results files: ')
        params['science_map'] = input('Please enter the science map to be considered: ')
        #params['constraint'] = input('Please give the value of the secondary constraint (either tau_obs or filter): ')
        params['output_dir'] = input('Please enter the path to the directory for output: ')
    else:
        params['sim_results_list'] = argv[1]
        params['metric'] = argv[2]
        params['science_map'] = argv[3]
        #params['constraint'] = argv[3]
        params['output_dir'] = argv[4]

    return params

def parse_sim_results_list(params):
    """The list of opsim results is a text file consisting of a list of entries
    of the form:
    opsim_identifier        path_to_results_file
    """

    if not path.isfile(params['sim_results_list']):
        raise IOError('Cannot find the list of OpSim results files')

    data_list = open(params['sim_results_list'],'r').readlines()

    params['sim_results'] = {}
    for line in data_list:
        entries = line.replace('\n','').split()
        params['sim_results'][entries[0]] = entries[1]

    return params

def parse_results_files(params):

    results_data = {}
    for opsim, results_file in params['sim_results'].items():

        if params['metric'] in FOOTPRINT_METRICS:
            dataset = plotfootprintFoM_barchart.parse_data_file(results_file)
        elif params['metric'] in CADENCE_METRICS:
            dataset = plotcadenceFoM_compare_opsims.parse_data_file_per_tau(results_file)
        elif params['metric'] in FILTER_METRICS:
            dataset = plot_timeperfilter_barchart.parse_data_file(results_file)
        elif params['metric'] in PERIOD_COLOR_METRICS:
            dataset = plot_period_color_metrics.parse_data_file(results_file)
        elif params['metric'] in ANNUAL_CADENCE_METRICS:
            dataset = plot_annual_cadence.parse_data_file(results_file)
        else:
            raise IOError('Unsupported metric ('+params['metric']+') requested')

        results_data[opsim] = dataset

    return results_data

def select_metric_data(params, results_data):

    metric_data = {}
    #print(results_data)

    for opsim, dataset in results_data.items():
        tau_values = []
        metric_values = []
        for tau in TAU_OBS:
            tau_data = dataset[str(tau)]
            idx = np.where(tau_data['maps'] == params['science_map'])
            tau_values.append(tau)
            metric_values.append(tau_data[idx][params['metric']])
        metric_data[opsim] = Table([Column(tau_values, name='tau', dtype=float),
                                    Column(metric_values, name='metric', dtype=float)])

    return metric_data

def plot_radial_chart(params,metric_data,ymax=100.0):

    (fig, ax) = plt.subplots(1, 1, figsize=(20, 10),
                                  subplot_kw={"projection": "polar"})
    #plt.subplots_adjust(left=0.05, right=0.9)

    # Set radial y-range according to the expected range of metric values.
    # In many cases, this is a percentage, so the default is 100%
    if params['metric'] == 'medianNYSOs':
        for opsim, dataset in metric_data.items():
            print(ymax, np.median(dataset['metric']))
            if np.median(dataset['metric']) > ymax:
                ymax = np.median(dataset['metric'])

    fig.patch.set_facecolor("white")

    ax.set_facecolor("white")
    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(0, ymax)

    plot_colours = ['#003f5c','#2f4b7c','#665191','#a05195',
                    '#d45087','#f95d6a','#ff7c43','#ffa600']
    icol = 0
    plot_symbols = ['$a$','$b$','$c$','$d$','$e$','$f$','$g$','$h$',
                    '$i$','$j$','$k$','$l$','$m$','$n$','$o$','$p$','$q$']
    isym = 0
    line_styles = ['-', '-.', '--', '..']
    iline = 0

    # Loop over each bandpass since radial plots are generated for all
    # science maps:
    for opsim, dataset in metric_data.items():

        # Values for the x axis
        theta = np.linspace(0.05, 2 * np.pi - 0.05, len(dataset), endpoint=False)

        # Plot the data
        ax.plot(theta, dataset['metric'], color=plot_colours[icol],
                linestyle=line_styles[iline], alpha=0.5, zorder=10, label=opsim)
        ax.scatter(theta, dataset['metric'], color=plot_colours[icol],
                    marker=plot_symbols[isym], s=120, zorder=10, label=opsim)

        icol += 1
        if icol == len(plot_colours):
            icol = 0
            iline += 1

        isym += 1
        if isym == len(plot_symbols):
            isym = 0

    # Label the bars
    labels = ["\n".join(wrap(c, 10, break_long_words=False)) for c in dataset['maps']]
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, size=18, color='grey');
    #ax.set_title('NVisits = '+str(nvisits)+', $\\tau_{obs}$='+str(tau_obs[nvisits])+'days')

    # Plot radial gridlines
    ax.vlines(theta, 0, ymax, color="#1f1f1f", ls=(0, (4, 4)), zorder=11)

    # Plot legend outside the plot face area
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * -0.001,
             box.width, box.height * 0.95])

    loc = (1.9, 0.4)
    l = ax.legend(loc='right', bbox_to_anchor=loc,
                    ncol=1, fontsize=16)

    l.legendHandles[0]._sizes = [50]
    if len(l.legendHandles) > 1:
        l.legendHandles[1]._sizes = [50]

    plt.tight_layout()
    plot_file = path.join(params['output_dir'],params['metric']+'_'+str(params['science_map'])+'_opsim_comparison_radial_plot.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close()

def plot_heat_map(params,metric_data,zmax=100.0):

    opsim_list = metric_data.keys()
#    for opsim, dataset in metric_data.items():
#        if opsim not in opsim_list:
#            opsim_list.append(opsim)

    # Note the data array is maps x opsims because the orientation
    # of the plotted image will have the opsims on the x-axis
    # The x-axis starts from the origin of 0 then increases towards the right
    # The y-axis starts from an origin of max then increases towards the upward
    data_grid = np.zeros( (len(TAU_OBS),len(opsim_list)) )

    for isim,opsim in enumerate(opsim_list):
        dataset = metric_data[opsim]
        for itau,tau in enumerate(TAU_OBS):
            idx = np.where(dataset['tau'] == tau)[0][0]
            data_grid[itau,isim] = dataset[idx]['metric']

    # The first + 1 increases the length
    xgrid = np.arange(0,len(opsim_list)+1,1)
    ygrid = np.arange(0,len(TAU_OBS)+1,1)

    norm = mpl.colors.Normalize(data_grid.min(), data_grid.max())

    fig, ax = plt.subplots(figsize=(len(opsim_list)+1,len(TAU_OBS)+1))
    ax.pcolormesh(xgrid, ygrid, data_grid, cmap="magma", norm=norm)
    ax.set_frame_on(False) # remove all spines

    ax.set_xticks(xgrid[0:-1]+0.5)
    ax.set_yticks(ygrid[0:-1]+0.5)
    ax.set_ylabel('$\\tau_{obs}$ [days]', fontsize=20)
    ax.set_xticklabels(opsim_list,rotation=45.0,horizontalalignment='right',fontsize=20)
    rev_tau = TAU_OBS
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
    cb.set_label(params['metric'], size=20)

    plt.tight_layout()
    plot_file = path.join(params['output_dir'],params['metric']+'_'+str(params['science_map'])+'_opsim_tau_comparison_heatmap.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close()

if __name__ == '__main__':
    params = get_args()
    plot_metric_radial_charts(params)
