from os import path
from sys import argv
import eval_survey_cadence

params = {}
if len(argv) == 1:
    params['opSim_list'] = input('Please enter the path to the list of OpSim databases: ')
    params['map_file_path'] = input('Please enter the path to the GP footprint map: ')
else:
    params['opSim_list'] = argv[1]
    params['map_file_path'] = argv[2]

if not path.isfile(params['opSim_list']):
    raise IOError('Cannot find list of opsims at '+params['opSim_list'])

opsim_db_list = open(params['opSim_list'],'r').readlines()

for db in opsim_db_list:
    db_path = db.replace('\n','')
    config = {'opSim_db_file': db_path, 'map_file_path': params['map_file_path']}
    eval_survey_cadence.run_metrics(config)
