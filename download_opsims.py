import requests
from os import path
from sys import argv
import wget

OPSIMS_DATABASE_URL = 'http://astro-lsst-01.astro.washington.edu:8080/'

if len(argv) == 1:
    selection = input('Please enter the selection string for database filenames, e.g. "_v2.0_10yrs.db": ')
    output_dir = input('Please enter the path to the output directory: ')
else:
    selection = argv[1]
    output_dir = argv[2]

index = requests.get(OPSIMS_DATABASE_URL)

for line in index.text.split('\n'):
    if selection in line:
        entries = line.split('"')
        db_url = path.join(OPSIMS_DATABASE_URL, entries[1])

        print('Downloading OpSim DB '+path.basename(db_url))

        local_file_name = path.join(output_dir,path.basename(db_url))
        if not path.isfile(local_file_name):
            local_file = wget.download(db_url, out=output_dir)
        else:
            print('...OpSim already downloaded, skipping.')
