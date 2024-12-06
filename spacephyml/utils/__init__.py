"""
Common utils used by multiple scripts
"""
import numpy as np
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
