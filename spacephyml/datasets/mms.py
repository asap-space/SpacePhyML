"""
Specific MMS Datasets.
"""

from os import makedirs

import numpy as np

from .general.mms import ExternalMMSData
from ..utils.file_download import missing_files, download_file_with_status
from ..transforms import Compose, Threshold, LogNorm, Roll

class MMS1IonDistLabeled(ExternalMMSData):
    """
    Examples:
        >>> from spacephyml.datasets import ExternalMMSData
        >>> dataset = ExternalMMSData('./mydataset.csv')

    Args:
        dataset (string):
            The dataset, either SCNov2017 or SCDec2017.
        path (string):
            The path for storing the dataset (not the actuall data).
        data_root (string):
            The override the default root directory to for the MMS data storage.
        transform (callable):
            Optional transform to be applied on each sample.
        cache (bool):
            If data should be cached.
        return_epoch (bool):
            If the label epoch should be returned.


    """

    _valid_datasets = ['SCNov2017', 'SCDec2017']
    _datasets = {'SCNov2017' : {
                    'url' : '',
                    'file' : 'dataset_nov_2017_clean.csv'
                },
                'SCDec2017' : {
                    'url' : '',
                    'file' : 'dataset_dec_2017_clean.csv'
                }}

    def __init__(self, dataset, path = './datasets', data_root = None,
                 transform = None, cache = True):
        if not dataset in self._valid_datasets:
            raise ValueError(f'Incorrect dataset, {dataset} not in {__valid_datasets}')

        filepath = f'{path}/' + self._datasets[dataset]['file']

        missing = missing_files([filepath],'./')
        if missing:
            print('Missing dataset file, downloading')
            makedirs(path, exist_ok=True)
            download_file_with_status(self._datasets[dataset]['url'], filepath)

        if transform == None:
            transform = Compose(Threshold((np.power(10.0,-28),np.power(10.0,-17))),
                                LogNorm((-28,-17)),
                                Roll())


        super().__init__(filepath, data_root, transform, cache)


