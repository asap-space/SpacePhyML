"""
Common utils used by multiple scripts
"""
from os import path
import numpy as np
import pandas as pd
import cdflib


def read_cdf_file(cdf_filepath, variables = None):
    """
    Read a cdf file, either fully or only a subset.

    Arguments:
        cdf_filepath (string): Path to the CDF file.
        variables (list): List with tuples the names to store
                    the varibles in and varibles to read.
    Output:
        Dictionary with the varibles.
    """
    if variables is None:
        return cdflib.cdfread.CDF(cdf_filepath)

    data = {}
    cdf_file = cdflib.cdfread.CDF(cdf_filepath)
    for name, var in variables:
        try:
            data[name] = np.array((cdf_file.varget(var)))
        except:
            print(f'Failed to read {var} from {cdf_filepath}')
            raise

    return data

def pandas_read_file(filepath):
    """
    Wrapper to handle reading data from multiple different fileformats.
    """
    _, fileformat = path.splitext(filepath)
    if fileformat == '.csv':
        return pd.read_csv(filepath)
    if fileformat == '.feather':
        return pd.read_feather(filepath)

    raise ValueError(f'Unknown filetype: {fileformat}')
