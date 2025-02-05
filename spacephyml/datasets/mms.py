"""
Module containing different datasets.
"""

from torch.utils.data import Dataset

import numpy as np

from ..utils import mms, read_cdf_file, pandas_read_file
from ..utils.file_download import missing_files
from ..__init__ import _MMS_DATA_DIR

class MMSDataset(Dataset):
    """
    Loading a dataset with labeled MMS data based on dataset file.

    By default SpacePhyML will look for MMS data at the PySPEDAS data location
    ([PySPEDAS](https://pyspedas.readthedocs.io/en/stable/getting_started.html#local-data-directories).)
    If the PySPEDAS environmental variable's are not set data will be placed at
    `$HOME/spacephyml_data/mms`, following the same directory structure as
    PySPEDAS (and the [MMS Science Data Center](https://lasp.colorado.edu/mms/sdc/public/)).
    Data files that are missing when the class is initialised will be downloaded.

    The dataset file have to have the following columns:

    - label : The label corresponding to the sample
    - epoch : The CDF epoch for the label
    - file {i} : Specifying the MMS CDF file to read data from, the {i} is a running number.
    - var_name {i} : The variable in the CDF file to read, the {i} is a running number.
    - epoch {i} : The CDF epoch to read data from the {i} is a running number.

    Note:
        If the loading data fail it may be due to the cdf file being corrupt. Delete
        the failing file and retry.

    Examples:
        >>> from spacephyml.datasets import MMSDataset
        >>> dataset = MMSDataset('./mydataset.csv')

    Args:
        dataset_path (string): Path to the file containing the dataset.
        rootdir (string): The override the default rootdir to for the MMS data storage.
        transform (callable): Optional transform to be applied on each sample.
        cache (bool): If data should be cached.
        return_epoch (bool): If the label epoch should be returned.

    Returns:
        Will return a list with with all the data varibles in a list followed by the label.
        If 'return_epoch = True' is set the the label epoch of the data will also be
        returned.

    """

    def __init__(self, dataset_path, rootdir = None, transform = None, cache = True,
                 return_epoch = True):

        self.dataset = pandas_read_file(dataset_path)
        self.cache = cache
        self.return_epoch = return_epoch

        if rootdir:
            self.rootdir = rootdir
        else:
            self.rootdir = _MMS_DATA_DIR

        # There are two extra columns and for each varible
        # there are three columns
        self.num_vars = int((len(self.dataset.columns)-2)/3)

        for i in range(self.num_vars):
            files = mms.filename_to_filepath(self.dataset[f'file {i}'].unique())
            missing = missing_files(files, self.rootdir)

            if missing:
                print(f"{len(missing)} data files are missing, downloading")
                mms.download_cdf_files(self.rootdir, missing)

            if self.cache:
                #Add an index for each entry
                self.dataset[f'index {i}'] = -1

        self.length = len(self.dataset.index)

        self.transform = transform

        self.data = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError('Expected idx to be an integer value')

        data_loc = self.dataset.iloc[idx,self.dataset.columns]

        sample = []
        for i in range(self.num_vars):
            cdf_filepath = f'{self.rootdir}/'
            cdf_filepath += f'{mms.filename_to_filepath(data_loc[f"file {i}"])}'

            data = {}
            if self.cache:
                if not data_loc[f'file {i}'] in self.data:
                     self.data[data_loc[f'file {i}']] = \
                        read_cdf_file(cdf_filepath,
                                      [('var',data_loc[f'var_name {i}']), ('epoch','epoch')])

                # This index caching does not seem to work
                index = data_loc[f'index {i}']
                if index == -1:
                    self.dataset.at[idx,f'index {i}'] =  \
                        np.where(self.data[data_loc[f'file {i}']]['epoch'] == data_loc[f'epoch {i}'])[0]
                    index = self.dataset.loc[idx,f'index {i}']

                sample.append(self.data[data_loc[f'file {i}']]['var'][index])
            else:
                data = read_cdf_file(cdf_filepath,
                                      [('var',data_loc[f'var_name {i}']), ('epoch','epoch')])

                index = np.where(
                        data['epoch'] == data_loc[f'epoch {i}'])
                sample.append(self.data[data_loc[f'file {i}']]['var'][index])

        sample.append(data_loc.label)

        if self.return_epoch:
            sample.append(data_loc.epoch)

        if self.transform:
            sample = self.transform(sample)

        return sample
