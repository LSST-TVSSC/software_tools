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

def parse_data_file(data_file):

    if not path.isfile(data_file):
        raise IOError('Cannot find input file '+data_file)

    file_lines = open(data_file, 'r').readlines()

    runName = []
    mapName = []
    nVisits_median = []
    nVisits_stddev = []
    shutterFrac_median = []
    shutterFrac_stddev = []

    for line in file_lines[1:]:
        if line[0:1] != '#':
            entries = line.replace('\n','').split()
            runName.append(entries[0])
            mapName.append(entries[1])
            nVisits_median.append(convert_value(entries[2],'float'))
            nVisits_stddev.append(convert_value(entries[3],'float'))
            shutterFrac_median.append(convert_value(entries[4],'float'))
            shutterFrac_stddev.append(convert_value(entries[5],'float'))

    dataset = Table([Column(runName, name='runName'),
                 Column(mapName, name='maps'),
                 Column(nVisits_median, name='nVisits_median', dtype=float),
                 Column(nVisits_stddev, name='nVisits_stddev', dtype=float),
                 Column(shutterFrac_median, name='shutterFrac_median', dtype=float),
                 Column(shutterFrac_stddev, name='shutterFrac_stddev', dtype=float)])

    print(dataset)
    return dataset

def convert_value(inVal,type):

    if 'none' in str(inVal).lower():
        return None

    try:
        if type == 'float':
            outVal = float(inVal)
        elif type == 'int':
            outVal = int(float(inVal))
        elif str in type:
            outVal = inVal
        else:
            outVal = inVal
        return outVal

    except ValueError:
        return None
