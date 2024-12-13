"""
Common utils used by multiple scripts
"""
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
        data[name] = np.array((cdf_file.varget(var)))

    return data

def pandas_read_file(filepath):
    _, fileformat = path.splittext(filepath)
    if fileformat == '.csv':
        return pd.read_csv(filepath)
    elif fileformat == '.feather':
        return pd.read_feather(filepath)
    else:
        raise ValueError(f'Unknown filetype: {fileformat}')
