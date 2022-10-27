from os import path
from sys import argv
import glob

def make_results_list(params):

    file_list = glob.glob(path.join(params['data_dir'],
                            '*'+params['search_key']+'*.txt'))

    data_list = {}
    for f in file_list:
        opsim = ((path.basename(f)).split(params['search_key'])[0])[0:-1]
        data_list[opsim] = f

    sorted = list(data_list.keys())
    sorted.sort()

    output = open(path.join(params['data_dir'],
                    'opsim_results_'+params['search_key']+'.txt'), 'w')
    for opsim in sorted:
        output.write(opsim+'  '+data_list[opsim]+'\n')
    output.close()

def get_args():

    params = {}

    if len(argv) == 1:
        params['data_dir'] = input('Please enter the path to the data directory: ')
        params['search_key'] = input('Please enter the search key used to select files [e.g. footprint, survey_cadence]: ')
    else:
        params['data_dir'] = argv[1]
        params['search_key'] = argv[2]

    return params


if __name__ == '__main__':
    params = get_args()
    make_results_list(params)
