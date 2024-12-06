"""
Module containing different datasets.
"""

from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from ..utils import mms, read_cdf_file
from ..utils.file_download import missing_files
from ..__init__ import _MMS_DATA_DIR

class MMSDataset(Dataset):
    """
    Loading a dataset with labeled MMS data based on dataset file.

    Examples:
        >>> from spacephyml.datasets import MMSDataset
        >>> dataset = MMSDataset('./mydataset.csv')

    Args:
        dataset_path (string): Path to the csv file containing the dataset.
        rootdir (string): The override the default rootdir to for the MMS data storage.
        transform (callable): Optional transform to be applied on each sample.
        cache (bool): If data should be cached.
    """

    def __init__(self, dataset_path, rootdir = None, transform = None, cache = True):

        self.dataset = pd.read_csv(dataset_path)
        self.cache = cache

        if rootdir:
            self.rootdir = rootdir
        else:
            self.rootdir = _MMS_DATA_DIR

        files = mms.filename_to_filepath(self.dataset['file'].unique())
        missing = missing_files(files, self.rootdir)

        if missing:
            print(f"{len(missing)} data files are missing, downloading")
            mms.download_cdf_files(self.rootdir, missing)

        self.length = len(self.dataset.index)

        self.transform = transform

        self.data = {}


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('Expected idx to be an integer value')

        data_loc = self.dataset.loc[idx,['file','var_name', 'epoch','label']]

        cdf_filepath = f'{self.rootdir}/'
        cdf_filepath += f'{mms.filename_to_filepath(data_loc.file)}'

        data = {}
        if self.cache:
            if not data_loc.file in self.data:
                tmp = read_cdf_file(cdf_filepath, [('var',data_loc.var_name), ('epoch','epoch')])
                file_epochs = self.dataset.loc[self.dataset['file']==data_loc.file,
                                                'epoch'].to_numpy()
                index = np.where(np.isin(tmp['epoch'],file_epochs))[0]
                self.data[data_loc.file] = {'var': tmp['var'][index],
                                            'epoch': tmp['epoch'][index]}
            data = self.data[data_loc.file]
        else:
            data = read_cdf_file(cdf_filepath, [('var',data_loc.var_name), ('epoch','epoch')])

        index = np.where(
                data['epoch'] == data_loc.epoch)

        sample = (
            data['var'][index][0],
            data_loc.label,
            data['epoch'][index][0],
            data_loc.file
        )

        if self.transform:
            sample = self.transform(sample)

        return sample
