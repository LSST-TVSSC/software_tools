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

def plot_data(params):

    (opsims, data_grid, columns) = read_data_file(params)

    plot_heatmap(params, opsims, data_grid, columns)

def plot_heatmap(params, opsims, data_grid, columns):

    # The first + 1 increases the length
    xgrid = np.arange(0,len(opsims)+1,1)
    ygrid = np.arange(0,len(columns)+1,1)

    # Normalize the color scheme used for the heatmap:
    norm = mpl.colors.Normalize(data_grid.min(), data_grid.max())

    # Adjust the overall size of the figure, depending on how many
    # parameters and opsims are included, and allow offsets from the
    # plot boundaries, to allow for the length of the labels
    figx = len(opsims)+2
    figy = len(columns)+2
    if len(opsims) <= 5:
        figx = 10
        figy = 10

    fig, ax = plt.subplots(figsize=(figx,figy))
    if len(opsims) <= 5:
        plt.subplots_adjust(left=0.35, bottom=0.35)

    # Plot the heatmap data
    ax.pcolormesh(xgrid, ygrid, data_grid, cmap="magma", norm=norm)
    ax.set_frame_on(False) # remove all spines

    ax.set_xticks(xgrid[0:-1]+0.5)
    ax.set_yticks(ygrid[0:-1]+0.5)
    ax.set_xticklabels(opsims,rotation=45.0,horizontalalignment='right',fontsize=20)
    ax.set_yticklabels(columns,fontsize=20,horizontalalignment='right')

    # Add a colorbar
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="magma"),
        #cax=cbar_ax, # Pass the new axis
        orientation = "vertical")
    cb.set_label(params['metric'], size=20)

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0)
    cb.ax.tick_params(labelsize=20)

    # Save to file:
    if len(opsims) >= 5:
        plt.tight_layout()
    print('Plot output to '+params['plot_file'])
    plt.savefig(params['plot_file'])
    plt.close()

def read_data_file(params):
    if not path.isfile(params['data_file']):
        raise IOError('Cannot find the input data file: '+params['data_file'])

    file_lines = open(params['data_file'],'r').readlines()

    data = []
    opsims = []
    columns = []
    for line in file_lines:
        if line[0:1] != '#':
            entries = line.replace('\n','').split()
            opsims.append(entries[0])
            row = []
            for value in entries[1:]:
                row.append(float(value))
            data.append(row)
        else:
            entries = line.replace('\n','').split(':')
            columns.append(entries[-1].strip())
    data_grid = np.array(data)

    # Data grid is transposed to match Python's image orientation:
    data_grid = np.transpose(data_grid)

    # Remove the first entry from the columns list because this is held
    # in the separate opsims list:
    columns = columns[1:]

    return opsims, data_grid, columns

def get_args():
    params = {}
    if len(argv) == 1:
        params['data_file'] = input('Please enter the path to the input datafile: ')
        params['plot_file'] = input('Please enter the path to the output plot file: ')
    else:
        params['data_file'] = argv[1]
        params['plot_file'] = argv[2]
    return params


if __name__ == '__main__':
    params = get_args()
    plot_data(params)
