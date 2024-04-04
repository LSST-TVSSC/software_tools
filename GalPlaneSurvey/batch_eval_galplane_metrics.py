from os import path
from sys import argv
import compare_survey_footprints
import eval_survey_cadence
import eval_time_per_filter
#import eval_period_color_metrics
import eval_annualVIM
import eval_survey_efficiency
import eval_microlensing_metric

metric_group_options = [
    'footprint',
    'cadence',
    'filter',
#    'period_color',
    'annual',
    'efficiency',
    'microlensing',
    'all'
]

params = {}
if len(argv) == 1:
    params['opSim_list'] = input('Please enter the path to the list of OpSim databases: ')
    params['map_file_path'] = input('Please enter the path to the GP footprint map: ')
    params['output_dir'] = input('Please enter the path to the directory for output: ')
    params['metric_groups'] = input('Please indicate which metric groups to evaluate {'
                                    + repr(metric_group_options) + '}: ')
else:
    params['opSim_list'] = argv[1]
    params['map_file_path'] = argv[2]
    params['output_dir'] = argv[3]
    params['metric_groups'] = argv[4]

if not path.isfile(params['opSim_list']):
    raise IOError('Cannot find list of opsims at '+params['opSim_list'])

params['metric_groups'] = params['metric_groups'].lower()
if params['metric_groups'] not in metric_group_options:
    raise IOError('Invalid option given for the metrics to be evaluated.')

opsim_db_list = open(params['opSim_list'],'r').readlines()

for db in opsim_db_list:
    db_path = db.replace('\n','')
    config = {'opSim_db_file': db_path,
              'map_file_path': params['map_file_path'],
              'output_dir': params['output_dir']}
    if params['metric_groups'] in ['cadence','all']:
        eval_survey_cadence.run_metrics(config)
    if params['metric_groups'] in ['footprint','all']:
        compare_survey_footprints.run_metrics(config)
    # Most of the metrics in this group seem to be depreciated
#    if params['metric_groups'] in ['period_color','all']:
#        eval_period_color_metrics.run_metrics(config)
    if params['metric_groups'] in ['filter','all']:
        eval_time_per_filter.run_metrics(config)
    if params['metric_groups'] in ['annual','all']:
        eval_annualVIM.run_metrics(config)
    if params['metric_groups'] in ['efficiency','all']:
        eval_survey_efficiency.run_metrics(config)
    if params['metric_groups'] in ['microlensing', 'all']:
        config['metric_calc'] = 'Npts'      # Either detect or Npts
        eval_microlensing_metric.run_metrics(config)
