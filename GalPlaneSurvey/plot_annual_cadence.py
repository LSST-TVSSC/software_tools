from os import path
from sys import argv
import numpy as np
from astropy.table import Table, Column
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap

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
YEARS_IN_SURVEY = 10

def plot_metric(params):

    # Parse the list of simulation results
    params = parse_sim_results_list(params)

    # Parse the datafiles, extracting the data on the requested metric, and
    # applying the requested constraint
    results_data = parse_results_files(params)
    metric_data = select_metric_data(params, results_data)

    # Generate a radial plot of the results
    #plot_radial_chart(params,metric_data)
    plot_heat_map_metric_all_years(params,metric_data,zmax=100.0)

def get_args():

    params = {}
    if len(argv) == 1:
        params['sim_results_list'] = input('Please enter the path to the list of OpSim results files: ')
        #params['metric'] = input('Please enter the identifier of the metric (column name) to extract from the results files: ')
        params['science_map'] = input('Please enter the science map to be considered: ')
        params['constraint'] = input('Please give the value of the tau constraint: ')
        params['output_dir'] = input('Please enter the path to the directory for output: ')
    else:
        params['sim_results_list'] = argv[1]
        #params['metric'] = argv[2]
        params['science_map'] = argv[2]
        params['constraint'] = argv[3]
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

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    dataset = {}

    for line in file_lines[1:]:
        if line[0:1] != '#':
            entries = line.replace('\n','').split()
            tau = float(entries[2])
            if str(tau) in dataset.keys():
                tau_data = dataset[str(tau)]
            else:
                tau_data = {'maps': [], 'runName': [], 'tau': [], 'tau_var': []}
                for iyear in range(1,YEARS_IN_SURVEY+1,1):
                    tau_data['vim_year'+str(iyear)] = []

            tau_data['runName'].append(entries[0])
            tau_data['maps'].append(entries[1].replace('_map',''))
            tau_data['tau'].append(float(entries[2]))
            tau_data['tau_var'].append(float(entries[3]))
            for iyear in range(0,YEARS_IN_SURVEY,1):
                tau_data['vim_year'+str(iyear+1)].append(float(entries[(4+iyear)]))

            dataset[str(tau)] = tau_data

    for tau, tau_data in dataset.items():
        col_list = [Column(tau_data['maps'], name='maps'),
                     Column(tau_data['runName'], name='runName'),
                     Column(tau_data['tau'], name='tau'),
                     Column(tau_data['tau_var'], name='tau_var')]
        for iyear in range(0,YEARS_IN_SURVEY,1):
            col_list.append( Column(tau_data['vim_year'+str(iyear+1)], name='vim_year'+str(iyear+1)) )

        dataset[tau] = Table(col_list)

    return dataset

def parse_results_files(params):

    results_data = {}
    for opsim, results_file in params['sim_results'].items():

        dataset = parse_data_file(results_file)

        results_data[opsim] = dataset

    return results_data

def select_metric_data(params, results_data):

    metric_data = {}

    for opsim, dataset in results_data.items():
        year_values = range(1,YEARS_IN_SURVEY+1,1)
        metric_values = []
        tau_data = dataset[params['constraint']]
        idx = np.where(tau_data['maps'] == params['science_map'])
        for iyear in range(1,YEARS_IN_SURVEY+1,1):
            metric_values.append(tau_data[idx]['vim_year'+str(iyear)])
        metric_data[opsim] = Table([Column(year_values, name='year', dtype=float),
                                    Column(metric_values, name='metric', dtype=float)])

    return metric_data

def plot_heat_map_metric_all_years(params,metric_data,zmax=100.0):

    opsim_list = metric_data.keys()
#    for opsim, dataset in metric_data.items():
#        if opsim not in opsim_list:
#            opsim_list.append(opsim)

    # Note the data array is maps x opsims because the orientation
    # of the plotted image will have the opsims on the x-axis
    # The x-axis starts from the origin of 0 then increases towards the right
    # The y-axis starts from the top then increases downwards
    data_grid = np.zeros( (YEARS_IN_SURVEY,len(opsim_list)) )
    for isim,opsim in enumerate(opsim_list):
        dataset = metric_data[opsim]
        for iyear in range(1,YEARS_IN_SURVEY+1,1):
            idx = np.where(dataset['year'] == float(iyear))[0][0]
            data_grid[iyear-1,isim] = dataset[idx]['metric']
    
    # The first + 1 increases the length
    xgrid = np.arange(0,len(opsim_list)+1,1)
    ygrid = np.arange(0,YEARS_IN_SURVEY+1,1)

    norm = mpl.colors.Normalize(0.0, 1.0)

    xmin = len(opsim_list)
    if xmin < 5: xmin = 10
    fig, ax = plt.subplots(figsize=(xmin,(YEARS_IN_SURVEY+1)))
    ax.pcolormesh(xgrid, ygrid, data_grid, cmap="magma", norm=norm)
    ax.set_frame_on(False) # remove all spines

    ax.set_xticks(xgrid[0:-1]+0.5)
    ax.set_yticks(ygrid[0:-1]+0.5)
    ax.set_ylabel('Year in survey', fontsize=20)
    ax.set_xticklabels(opsim_list,rotation=45.0,horizontalalignment='right',fontsize=20)
    ax.set_yticklabels(range(1,YEARS_IN_SURVEY+1,1),fontsize=20,horizontalalignment='right')

    tvar = float(params['constraint'])*5.0
    ax.set_title('VisitIntervalMetric for '+str(params['science_map'])+', $\\tau_{var}=$'+str(tvar), fontsize=20)
    # Remove ticks by setting their length to 0
    #ax.yaxis.set_tick_params(length=0)
    #ax.xaxis.set_tick_params(length=0)

    # Add a colorbar scale, coords=(x coordinate of left border,
    #  y coordinate for bottom border,
    #  width,
    #  height)
    #fig.subplots_adjust(bottom=0.25)
    #cbar_ax = fig.add_axes([0.3, -0.05, 0.4, 0.025])

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="magma"),
        #cax=cbar_ax, # Pass the new axis
        orientation = "vertical")

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0)
    cb.ax.tick_params(labelsize=20)

    # Set legend label
    cb.set_label('vim', size=20)

    plt.tight_layout()
    plot_file = path.join(params['output_dir'],'vim_'+str(params['science_map'])+'_'+str(params['constraint'])+'_yearly_comparison_heatmap.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close()

if __name__ == '__main__':
    params = get_args()
    plot_metric(params)
