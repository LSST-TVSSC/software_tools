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

    plot_radial_barchart(dataset)

def plot_radial_barchart(dataset):
    "Based on example from Python Graph Gallery"

    # The tau_obs values corresponding to the categories of NVisits
    tau_obs = {7200: 2.0,
                2880: 5.0,
                1309: 11.0,
                720: 20.0,
                310: 46.5,
                197: 73.0}
    plot_order = [197, 310, 720, 1309, 2880, 7200]

    # Plot configuration
    COLORS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)
    #norm = mpl.colors.Normalize(vmin=0, vmax=15)
    #COLORS = cmap(norm(15))
    plt.rcParams['text.color'] = "#1f1f1f"
    plt.rcParams['font.size'] = 22

    nrows = 3
    ncols = 2
    (fig, ax) = plt.subplots(nrows,ncols, figsize=(20, 20),
                                  subplot_kw={"projection": "polar"})
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    fig.patch.set_facecolor("white")

    irow = 0
    icol = -1
    for nvisits in plot_order:
        tau_data = dataset[nvisits]
        icol+=1
        if icol >= ncols:
            icol = 0
            irow+=1

        # Values for the x axis
        theta = np.linspace(0.05, 2 * np.pi - 0.05, len(tau_data), endpoint=False)

        ax[irow,icol].set_facecolor("white")
        ax[irow,icol].set_theta_offset(1.2 * np.pi / 2)
        ax[irow,icol].set_ylim(0, 100)

        # Plot the data
        ax[irow,icol].bar(theta, tau_data['%overlap'], color=COLORS, alpha=0.75, width=0.52, zorder=10)

        # Plot radial gridlines
        ax[irow,icol].vlines(theta, 0, 100, color="#1f1f1f", ls=(0, (4, 4)), zorder=11)

        # Label the bars
        labels = ["\n".join(wrap(c, 10, break_long_words=False)) for c in tau_data['maps']]
        ax[irow,icol].set_xticks(theta)
        ax[irow,icol].set_xticklabels(labels, size=22);
        ax[irow, icol].set_title('NVisits = '+str(nvisits)+', $\\tau_{obs}$='+str(tau_obs[nvisits])+'days')

    plt.tight_layout()
    plt.savefig(path.join('results',tau_data['runName'][0]+'_field_overlap_radial_plot.png'))
    plt.close()

    (fig, ax) = plt.subplots(1,1, figsize=(10, 10),
                                  subplot_kw={"projection": "polar"})
    #plt.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.patch.set_facecolor("white")
    nvisits = 720
    tau_data = dataset[nvisits]

    theta = np.linspace(0.05, 2 * np.pi - 0.05, len(tau_data), endpoint=False)

    ax.set_facecolor("white")
    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(0, 100)

    # Plot the data
    ax.bar(theta, tau_data['%overlap'], color=COLORS, alpha=0.75, width=0.52, zorder=10)

    # Plot radial gridlines
    ax.vlines(theta, 0, 100, color="#1f1f1f", ls=(0, (4, 4)), zorder=11)

    # Label the bars
    labels = ["\n".join(wrap(c, 10, break_long_words=False)) for c in tau_data['maps']]
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, size=22);
    ax.set_title('NVisits = '+str(nvisits)+', $\\tau_{obs}$='+str(tau_obs[nvisits])+'days')

    plt.tight_layout()
    plt.savefig(path.join('results',tau_data['runName'][0]+'_field_overlap_radial_plot_'+str(nvisits)+'.png'))
    plt.close()

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    dataset = {}

    for line in file_lines[1:]:
        entries = line.replace('\n','').split()
        nvisits = int(float(entries[3]))
        mapName = entries[1]
        runName = entries[0]
        percent = float(entries[5])
        if nvisits in dataset.keys():
            tau_data = dataset[nvisits]
        else:
            tau_data = {'maps': [], 'runName': [], 'data': []}
        tau_data['maps'].append(mapName.replace('_map',''))
        tau_data['runName'].append(runName)
        tau_data['data'].append(percent)
        dataset[nvisits] = tau_data

    for nvisits, tau_data in dataset.items():
        tab = Table([Column(tau_data['data'], name='%overlap', dtype=float),
                     Column(tau_data['runName'], name='runName'),
                    Column(tau_data['maps'], name='maps')])
        dataset[nvisits] = tab

    return dataset


if __name__ == '__main__':

    args = {}
    if len(argv) == 1:
        args['data_file'] = input('Please enter the path to the data file: ')
    else:
        args['data_file'] = argv[1]

    plot(args)
