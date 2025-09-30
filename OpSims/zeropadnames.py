from os import path, rename
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='Path to data directory')
parser.add_argument('rootname', help='Root name of files to process')
args = parser.parse_args()

def padd_filename(args, filepath, nchar):

    idx = path.basename(filepath).replace(args.rootname + '_', '').replace('.png', '')

    while len(idx) < nchar:
        idx = '0' + idx

    new_filepath = path.join(args.data_dir, 'frame_' + idx + '.png')

    if new_filepath != filepath:
        rename(filepath, new_filepath)
        print(filepath + ' -> ' + new_filepath)

    return new_filepath

file_list = glob.glob(path.join(args.data_dir, args.rootname + '*png'))

for filepath in file_list:
    new_filepath = padd_filename(args, filepath, 4)
