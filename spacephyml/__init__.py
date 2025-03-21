"""
SpacePhyML
"""
from os import environ

__version__ = '0.1.0'
__author__ = 'Jonah Ekelund'
__credits__ = ''

_HOME = '~'
if 'HOME' in environ:
    _HOME = environ['HOME']

_MMS_DATA_DIR = f'{_HOME}/spacephyml_data/mms/'

# Check for PySpedas directories
if 'MMS_DATA_DIR' in environ:
    _MMS_DATA_DIR = environ['MMS_DATA_DIR']
elif 'SPEDAS_DATA_DIR' in environ:
    _MMS_DATA_DIR = environ['SPEDAS_DATA_DIR']
