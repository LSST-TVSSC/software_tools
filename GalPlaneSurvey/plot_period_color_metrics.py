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
    tau = []
    tau_var = []
    percent_periodDetect = []
    medianUniformity = []
    medianNYSOs = []
    medianfTgaps = []

    for line in file_lines[1:]:
        if line[0:1] != '#':
            entries = line.replace('\n','').split()
            runName.append(entries[0])
            mapName.append(entries[1])
            tau.append(convert_value(entries[2],'float'))
            tau_var.append(convert_value(entries[3],'float'))
            percent_periodDetect.append(convert_value(entries[4],'float'))
            medianUniformity.append(convert_value(entries[5],'float'))
            medianNYSOs.append(convert_value(entries[6],'float'))
            medianfTgaps.append(convert_value(entries[7],'float'))

    dataset = Table([Column(runName, name='runName'),
                 Column(mapName, name='maps'),
                 Column(tau, name='tau'),
                 Column(tau_var, name='tau_var', dtype=float),
                 Column(percent_periodDetect, name='percent_periodDetect', dtype=float),
                 Column(medianNYSOs, name='medianNYSOs', dtype=float),
                 Column(medianfTgaps, name='medianfTgaps', dtype=float)])

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
