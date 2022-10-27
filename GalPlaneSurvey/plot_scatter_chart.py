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

    datasets = read_data_files(params)

    plot_scatter_chart(params, datasets)

def plot_scatter_chart(params, datasets):

    # Configuration of plot
    plot_colours = ['m', 'g', 'c']
    plot_symbols = ['d', 'x', 's']

    iplt = -1
    for datalabel, data_list in datasets.items():
        (opsims, data_grid, columns) = data_list
        iplt += 1

        if iplt == 0:
            # Adjust the overall size of the figure, depending on how many
            # parameters and opsims are included, and allow offsets from the
            # plot boundaries, to allow for the length of the labels
            figx = len(opsims)+2
            figy = len(columns)+2
            if len(opsims) <= 5:
                figx = 10
                figy = 10

            fig, ax = plt.subplots(figsize=(figx,figy))
            if len(opsims) <= 10:
                plt.subplots_adjust(left=0.25, bottom=0.45)

        # Plot the scatter chart:
        xvalues = range(0,len(opsims),1)
        col = plot_colours[iplt]
        sym = plot_symbols[iplt]
        ax.plot(xvalues, data_grid[:,params['column']], col+sym, label=datalabel)
        ax.plot(xvalues, data_grid[:,params['column']], col+'-')

        if iplt == 0:
            fontsize = 16.0
            plt.xticks(xvalues, labels=opsims,
                        rotation=45.0, ha='right', fontsize=fontsize)
            plt.tick_params(labelsize=20)
            plt.xlabel('OpSim runName', fontsize=fontsize)
            plt.ylabel(params['ylabel'], fontsize=fontsize)
            plt.grid()

    # Add legend:
    plt.legend()
    # Save to file:
    if len(opsims) >= 5:
        plt.tight_layout()
    print('Plot output to '+params['plot_file'])
    plt.savefig(params['plot_file'])
    plt.close()

def read_data_files(params):

    datasets = {}

    for n in range(1,3,1):
        datakey = 'data_file'+str(n)
        labelkey = params['dataset'+str(n)]

        if 'none' not in params[datakey].lower():
            if not path.isfile(params[datakey]):
                raise IOError('Cannot find the input data file: '+params[datakey])

            file_lines = open(params[datakey],'r').readlines()

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

            # Remove the first entry from the columns list because this is held
            # in the separate opsims list:
            columns = columns[1:]

            datasets[labelkey] = [opsims, data_grid, columns]

    return datasets

def get_args():
    params = {}
    if len(argv) == 1:
        params['data_file1'] = input('Please enter the path to the input datafile 1: ')
        params['dataset1'] = input('Please enter the dataset label for datafile 1: ')
        params['data_file2'] = input('Please enter the path to the input datafile 2 [or None]: ')
        params['dataset2'] = input('Please enter the dataset label for datafile 2 [or None]: ')
        params['plot_file'] = input('Please enter the path to the output plot file: ')
        params['column'] = input('Please enter the column to plot, indexed from zero not including runName column: ')
        params['ylabel'] = input('Name of the metric label: ')
    else:
        params['data_file1'] = argv[1]
        params['dataset1'] = argv[2]
        params['data_file2'] = argv[3]
        params['dataset2'] = argv[4]
        params['plot_file'] = argv[5]
        params['column'] = argv[6]
        params['ylabel'] = argv[7]
    params['column'] = int(float(params['column']))
    return params


if __name__ == '__main__':
    params = get_args()
    plot_data(params)
