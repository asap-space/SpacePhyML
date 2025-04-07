"""
Specific MMS Datasets.
"""

from os import makedirs

import numpy as np

from .general.mms import ExternalMMSData
from ..utils.file_download import missing_files, download_file_with_status
from ..transforms import MMS1IonDistLabeled_Transform


class MMS1IonDistLabeled(ExternalMMSData):
    """

    Examples:
        >>> from spacephyml.datasets.mms import MMS1IonDistLabeled
        >>> dataset = MMS1IonDistLabeled('SCDec017')

    [^1]: Olshevsky, V., et al. (2021). Automated classification of plasma regions using 3D particle energy distributions. Journal of Geophysical Research: Space Physics, https://doi.org/10.1029/2021JA029620
    """

    _valid_datasets = ['SCNov2017', 'SCDec2017']
    _datasets = {'SCNov2017': {
                    'url': 'https://zenodo.org/records/15147451/files/dataset_nov_2017_clean.csv?download=1',
                    'file': 'dataset_nov_2017_clean.csv'
                },
                'SCDec2017': {
                    'url': 'https://zenodo.org/records/15147451/files/dataset_dec_2017_clean.csv?download=1',
                    'file': 'dataset_dec_2017_clean.csv'
                }}

    def __init__(self, dataset, path='./datasets', data_root=None,
                 transform=None, cache=True, return_epoch = False):
        """

        Args:
            dataset (string):
                The dataset, either SCNov2017 or SCDec2017.
            path (string):
                The path for storing the dataset (not the actuall data).
            data_root (string):
                The override the default root directory to for the MMS data
                storage.
            transform (callable):
                Optional transform to be applied on each sample.
            cache (bool):
                If data should be cached.
            return_epoch (bool):
                If the label epoch should be returned.
        """
        if dataset not in self._valid_datasets:
            raise ValueError(f'Incorrect dataset, {dataset} not in' +
                             '{self._valid_datasets}')

        filepath = f'{path}/' + self._datasets[dataset]['file']

        missing = missing_files([filepath], './')
        if missing:
            print('Missing dataset file, downloading')
            makedirs(path, exist_ok=True)
            download_file_with_status(self._datasets[dataset]['url'], filepath)

        if transform is None:
            transform = MMS1IonDistLabeled_Transform()

        super().__init__(filepath, data_root, transform, cache, return_epoch)
