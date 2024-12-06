import cdflib
import numpy as np

from os import path

def read_cdf_file(cdf_filepath, vars = None):
    if vars == None:
        return cdflib.cdfread.CDF(cdf_filepath)

    data = {}
    cdf_file = cdflib.cdfread.CDF(cdf_filepath)
    for name, var in vars:
        data[name] = np.array((cdf_file.varget(var)))

    return data
