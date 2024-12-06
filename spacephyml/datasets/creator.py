"""
Script for creating dataset based on exisiting labels.
"""
import tempfile
from os import path, makedirs, remove
from datetime import datetime

import pandas as pd

from ..utils import read_cdf_file
from ..utils.file_download import download_file_with_status

_LABELS_URL_BASE = 'https://bitbucket.org/volshevsky/mmslearning/' + \
                   'raw/7b93d08b585842454c309668870ecd25ea16e3e0/labels_human/'
_LABELS_FILENAME_BASE = 'labels_fpi_fast_dis_dist_'

def download_label_file(outputpath, filedate):
    """
    Download Olshevsky label files.
    """
    makedirs(outputpath, exist_ok=True)
    file = _LABELS_FILENAME_BASE + filedate + '.cdf'
    download_file_with_status(_LABELS_URL_BASE + file, outputpath + file)
    return outputpath + file

def get_olshevsky_label_list(trange = None):
    """
    Get a pandoc DataFrame containing all the Olshevsky labels from within the given
    time range.
    """
    if trange is None:
        trange = [datetime(2017,11,1),datetime(2017,12,31)]
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

def get_dataset(label_source = 'Olshevsky', trange = None,
                       sampled = True, samples_per_label = 5000, clean = False):
    """
    Get a dataset based on a given config.
    """

    if trange is None:
        trange = ['2017-11-01', '2017-12-31']

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

def create_dataset(dataset_path, trange, force = False, sampled = True, clean = True):
    """
    Create a dataset file based on given config.
    """

    dirpath, _ = path.split(path.abspath(dataset_path))
    makedirs(dirpath, exist_ok=True)

    if path.isfile(dataset_path):
        if force:
            remove(dataset_path)
        else:
            print("Dataset exists, aborting")
            return

    labels = get_dataset(trange=trange, sampled=sampled, clean=clean)

    labels.to_csv(dataset_path)
