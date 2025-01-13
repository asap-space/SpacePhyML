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

def _download_label_file(outputpath, filedate):
    """
    Download Olshevsky label files.
    """
    makedirs(outputpath, exist_ok=True)
    file = _LABELS_FILENAME_BASE + filedate + '.cdf'
    download_file_with_status(_LABELS_URL_BASE + file, outputpath + file)
    return outputpath + file

_OLSHEVSKY_REF = """
Olshevsky, V., et al. (2021). Automated classification of plasma
regions using 3D particle energy distributions. Journal of Geophysical Research:
Space Physics, https://doi.org/10.1029/2021JA029620
"""

def _get_olshevsky_label_list(trange = None):
    """
    Get a pandoc DataFrame containing all the Olshevsky labels from within the given
    time range.
    """
    if trange is None:
        trange = [datetime(2017,11,1),datetime(2017,12,31)]
    elif trange[0] < datetime(2017,11,1) or datetime(2017,12,31) < trange[1]:
        raise ValueError('Invalid time range: range have to be in the range 2017-11-01 ' + \
                         'to 2017-12-31, (inclusive)')

    #Download the labelfiles from Olshevsky
    print('Downloading Olshevsky label files.')
    label_files = [_download_label_file(tempfile.gettempdir() + \
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
            #TODO: Here the data level and file version is hardcoded,
            # we could dynamicaly set this.
            file = [l[6:20] + 'l2_' + l[20:23] + '-' + l[24:] + '_v3.4.0.cdf'
                        for _ in range(len(label))]
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

def _get_unlabeled_dataset(trange):
    """
    Get a list of data in a given timerange.
    """
    print(trange)
    raise NotImplementedError

def get_dataset(label_source, trange, clean = True, samples = 0):
    """
    Get a dataset based on a given config.
    """

    trange = [datetime.strptime(d,'%Y-%m-%d') for d in trange]

    if label_source == 'Olshevsky':
        print('Generating a mms dataset based on labels from ')
        print(f'\t{_OLSHEVSKY_REF}')
        dataset = _get_olshevsky_label_list(trange)
        dataset['var_name'] = 'mms1_dis_dist_fast'
    elif label_source == 'Unlabeled':
        dataset = _get_unlabeled_dataset(trange)
    else:
        raise ValueError(f'Incorrect label_source ({label_source})')

    print('Creating dataset based on labels')
    if clean:
        dataset = dataset.loc[dataset['label'] != -1]

    if samples > 0:
        dataset = dataset.groupby('label').sample(n=samples)

        #Check that we actually have enought data here.
        for label in dataset['label'].unique():
            if len(dataset.loc[dataset['label'] == label]) < samples:
                raise ValueError('Not enought samles to create data set')

    dataset.sort_values('epoch', inplace=True)
    dataset.reset_index(drop=True, inplace = True)

    return dataset

def create_dataset(dataset_path, label_source, trange,
                   force = False, **kwargs):
    """
    Create a dataset file based on given config.
    """

    dataset_path = path.abspath(dataset_path)
    dirpath, _ = path.split(dataset_path)
    makedirs(dirpath, exist_ok=True)

    if path.isfile(dataset_path):
        if force:
            remove(dataset_path)
        else:
            print("Dataset exists, aborting")
            return

    labels = get_dataset(label_source, trange=trange, **kwargs)

    print(f'Storing dataset at {dataset_path}')
    _, fileformat = path.splitext(dataset_path)
    if fileformat == '.csv':
        labels.to_csv(dataset_path)
    elif fileformat == '.feather':
        labels.to_feather(dataset_path)
    else:
        raise ValueError(f'Unknown filetype {fileformat}')
