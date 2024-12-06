import sys
from os import path, listdir, makedirs, remove
import pprint
import pandas as pd
import itertools
from argparse import ArgumentParser
from datetime import datetime

from utils import read_cdf_file
from utils.file_download import missing_files, download_file_with_status
import tempfile

_LABELS_URL_BASE = 'https://bitbucket.org/volshevsky/mmslearning/raw/7b93d08b585842454c309668870ecd25ea16e3e0/labels_human/'
_LABELS_FILENAME_BASE = 'labels_fpi_fast_dis_dist_'

def download_label_file(outputpath, filedate):
    makedirs(outputpath, exist_ok=True)
    file = _LABELS_FILENAME_BASE + filedate + '.cdf'
    download_file_with_status(_LABELS_URL_BASE + file, outputpath + file)
    return outputpath + file

def get_olshevsky_label_list(trange = [datetime(2017,11,1),datetime(2017,12,31)]):
    #Download the labelfiles from Olshevsky
    print('Downloading Olshevsky label files.')
    label_files = [download_label_file(tempfile.gettempdir() + \
                    '/mms_labels/', d) for d in ['201711', '201712']]

    data = {'label':[],'epoch':[],'file':[], 'date':[]}
    for file in label_files:
        print(f'Processing file {file}')
        cdf_file = read_cdf_file(file)
        labels_vars = cdf_file.cdf_info().zVariables
        labels_vars = zip([ l for l in labels_vars if 'label_' in l],
                          [ e for e in labels_vars if 'epoch_' in e])

        for l, e in labels_vars:
            #Double check that the labels and epochs are refering to the same cdf file.
            if l[-14:] != e[-14:]:
                raise ValueError('The label and epoch vals are not equal')

            label = cdf_file.varget(l)
            epoch = cdf_file.varget(e)
            #TODO: Here the file version is hardcoded, we could dynamicaly check this
            file = [l[6:] + '_v3.4.0.cdf' for _ in range(len(label))]
            date = [datetime.strptime(l.split('_')[6][:8],'%Y%m%d')
                    for _ in range(len(label))]

            data['file'].extend(file)
            data['date'].extend(date)
            data['label'].extend(label)
            data['epoch'].extend(epoch)

    data = pd.DataFrame(data)
    data = data.loc[(trange[0] <= data['date']) &
                    (data['date'] <= trange[1])]

    return data.reset_index(drop=True).drop(columns=['date'])

def get_even_data_dist(label_source = 'Olshevsky', trange = ['2017-11-01', '2017-12-31'], sampled = True, samples_per_label = 5000, clean = False):

    trange = [datetime.strptime(d,'%Y-%m-%d') for d in trange]

    if label_source == 'Olshevsky':
        labels = get_olshevsky_label_list(trange)
        labels['var_name'] = 'mms1_dis_dist_fast'

    print('Creating dataset based on labels')
    if clean:
        labels = labels.loc[labels['label'] != -1]

    if sampled:
        labels = labels.groupby('label').sample(n=samples_per_label)
        #Change that we actually have enought data here.
        for label in labels['label'].unique():
            if len(labels.loc[labels['label'] == label]) < samples_per_label:
                raise ValueError('Not enought samles to create data set')

    labels.reset_index(drop=True, inplace = True)

    return labels

def create_dataset(dataset_path, trange, sampled = True, clean = True):
    dirpath, filename = path.split(path.abspath(dataset_path))
    makedirs(dirpath, exist_ok=True)

    if path.isfile(dataset_path):
        if args.force:
            remove(dataset_path)
        else:
            print("Dataset exists, aborting")
            return

    labels = get_even_data_dist(trange=trange, sampled=sampled, clean=clean)

    labels.to_csv(dataset_path)

def pars_args():
    parser = ArgumentParser()

    parser.add_argument('--config', default=None, choices=['orbit', 'orbitclean', 'nov', 'nov_full','dec'])
    parser.add_argument('--start', default='2017-11-01')
    parser.add_argument('--end', default='2017-11-31')
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--samples', default=0)
    parser.add_argument('output', nargs='?', default=None)

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f" {arg}: {getattr(args,arg)}")

    return args

if __name__ == "__main__":
    args = pars_args()

    labels_dl_dir = tempfile.gettempdir() + '/mms_labels/'

    year = 2017
    sampled = False
    clean = False
    if args.trange == 'orbit':
        month = 12
        trange=['2017-12-03','2017-12-07']
        sampled = False
    elif args.trange == 'orbitclean':
        month = 12
        trange=['2017-12-03','2017-12-07']
        clean = True
        sampled = False
    elif args.trange == 'dec':
        clean = True
        sampled = True
        month = 12
        trange=['2017-12-01','2017-12-31']
    elif args.trange == 'nov_full':
        clean = False
        sampled = False
        month = 11
        trange=['2017-11-01','2017-11-30']
    else:
        clean = True
        sampled = True
        month = 11
        trange=['2017-11-01','2017-11-30']

    if args.output == None:
        args.output = f'./dataset_{args.trange}_{year}_{month}'
        args.output += '_clean.csv' if clean else '.csv'

    create_dataset(args.output, trange, sampled, clean)
