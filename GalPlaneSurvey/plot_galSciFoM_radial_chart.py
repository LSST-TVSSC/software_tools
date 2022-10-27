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

tau_obs = {7200: 2.0,
            2880: 5.0,
            1309: 11.0,
            720: 20.0,
            310: 46.5,
            197: 73.0}
plot_order = [197, 310, 720, 1309, 2880, 7200]

output_dir = './FoM_results'

# Plot configuration
COLORS = ["m","orange"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)
#norm = mpl.colors.Normalize(vmin=0, vmax=15)
#COLORS = cmap(norm(15))
plt.rcParams['text.color'] = "#1f1f1f"
plt.rcParams['font.size'] = 22

def plot(args):

    dataset1 = parse_data_file(args['data_file1'])
    if 'None' not in args['data_file2']:
        dataset2 = parse_data_file(args['data_file2'])

    nrows = 3
    ncols = 2
    (fig, ax) = plt.subplots(nrows,ncols, figsize=(20, 20),
                                  subplot_kw={"projection": "polar"})
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    fig.patch.set_facecolor("white")

    irow = 0
    icol = -1
    ymax = 6.0
    for nvisits in plot_order:
        icol+=1
        if icol >= ncols:
            icol = 0
            irow+=1

        ax[irow,icol].set_facecolor("white")
        ax[irow,icol].set_theta_offset(1.2 * np.pi / 2)
        ax[irow,icol].set_ylim(0, ymax)

        tau_data = getMapData(dataset1, tau_obs[nvisits])

        # Values for the x axis
        theta = np.linspace(0.05, 2 * np.pi - 0.05, len(tau_data), endpoint=False)

        # Plot the data
        ax[irow,icol].plot(theta, tau_data['galSciFoM'], color=COLORS[0], linestyle='-', alpha=0.5, zorder=10)
        ax[irow,icol].scatter(theta, tau_data['galSciFoM'], color=COLORS[0], s=60, zorder=10)

        # Label the bars
        labels = ["\n".join(wrap(c, 10, break_long_words=False)) for c in tau_data['maps']]
        ax[irow,icol].set_xticks(theta)
        ax[irow,icol].set_xticklabels(labels, size=22);
        ax[irow, icol].set_title('NVisits = '+str(nvisits)+', $\\tau_{obs}$='+str(tau_obs[nvisits])+'days')

        # Plot radial gridlines
        ax[irow,icol].vlines(theta, 0, ymax, color="#1f1f1f", ls=(0, (4, 4)), zorder=11)

        if 'None' not in args['data_file2']:
            tau_data2 = getMapData(dataset2, tau_obs[nvisits])

            # Values for the x axis
            theta = np.linspace(0.05, 2 * np.pi - 0.05, len(tau_data2), endpoint=False)

            # Plot the data
            ax[irow,icol].plot(theta, tau_data2['galSciFoM'], color=COLORS[1], linestyle='-', alpha=0.75, zorder=10)
            ax[irow,icol].scatter(theta, tau_data2['galSciFoM'], color=COLORS[1], s=60, zorder=10)

    plt.tight_layout()
    print(args['data_file2'], ('None' in args['data_file2']))
    if 'None' in args['data_file2']:
        plot_file = path.join(output_dir,dataset1['runName'][0]+'_galSciFoM_radial_plot.png')
    else:
        plot_file = path.join(output_dir,dataset1['runName'][0]+'_'+dataset2['runName'][0]+'_galSciFoM_radial_plot.png')
    print('Plot output to '+plot_file)
    plt.savefig(plot_file)
    plt.close()


def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    runName = []
    mapName = []
    tau_obs = []
    bandpass = []
    footprint_priority = []
    ideal_footprint_priority = []
    region_priority_percent = []
    sumVIM = []
    percent_sumVIM = []
    mean_fexpt_ratio = []
    stddev_fexpt_ratio = []
    pix_obs = []
    npix_percent = []
    galSciFoM = []

    for line in file_lines[1:]:
        entries = line.replace('\n','').split()
        runName.append(entries[0])
        mapName.append(entries[1])
        tau_obs.append(entries[2])
        bandpass.append(entries[3])
        footprint_priority.append(entries[4])
        ideal_footprint_priority.append(entries[5])
        region_priority_percent.append(entries[6])
        sumVIM.append(entries[7])
        percent_sumVIM.append(entries[8])
        mean_fexpt_ratio.append(entries[9])
        stddev_fexpt_ratio.append(entries[10])
        pix_obs.append(entries[11])
        npix_percent.append(entries[12])
        galSciFoM.append(entries[13])

    dataset = Table([Column(runName, name='runName'),
                 Column(mapName, name='maps'),
                 Column(tau_obs, name='tau_obs', dtype=float),
                 Column(bandpass, name='bandpass'),
                 Column(footprint_priority, name='footprint_priority', dtype=float),
                 Column(ideal_footprint_priority, name='ideal_footprint_priority', dtype=float),
                 Column(region_priority_percent, name='region_priority_percent', dtype=float),
                 Column(sumVIM, name='sumVIM', dtype=float),
                 Column(percent_sumVIM, name='percent_sumVIM', dtype=float),
                 Column(mean_fexpt_ratio, name='mean_fexpt_ratio', dtype=float),
                 Column(stddev_fexpt_ratio, name='stddev_fexpt_ratio', dtype=float),
                 Column(pix_obs, name='pix_obs', dtype=float),
                 Column(npix_percent, name='npix_percent', dtype=float),
                 Column(galSciFoM, name='galSciFoM', dtype=float)])

    print(dataset)
    return dataset

def getMapData(dataset, tau_obs):

    # Get a list of the science maps in the dataset
    scienceMaps = []
    for map in dataset['maps']:
        if map not in scienceMaps:
            scienceMaps.append(map)

    # Fet the galSciFoM values for each science map for this run,
    # taking the first entry in each case because it is duplicated for
    # different filters and tau_obs values:
    dataFoM = []
    for map in scienceMaps:
        FoM_per_filter = dataset['galSciFoM'][(dataset['maps'] == map) & (dataset['tau_obs'] == tau_obs)]
        dataFoM.append(FoM_per_filter.sum())

    return Table([Column(scienceMaps, name='maps'),
                  Column(dataFoM, name='galSciFoM', dtype=float)])

if __name__ == '__main__':

    args = {}
    if len(argv) == 1:
        args['data_file1'] = input('Please enter the path to the data file: ')
        args['data_file2'] = input('Please enter the path to the data file: ')
    else:
        args['data_file1'] = argv[1]
        args['data_file2'] = argv[2]

    plot(args)
