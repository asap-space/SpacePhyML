"""
Dataset utils specific to MMS.
"""
from os import path, makedirs
import requests

from .file_download import download_file_with_status

def filename_to_filepath(filename):
    """
    Create a filepath based on the filename for a mms CDF file

    Filename format is assumed to be:
        mms1_fpi_fast_l2_dis-dist_20171129160000_v3.4.0.cdf
        {probe}_{inst}_{mode}_{data_level}_{data}_{time}_{version}.cdf
    Output will have the format:
    ./mms/mms1/fpi/fast/l2/dis-dist/2017/12/{filename}
    """
    if isinstance(filename, str):
        filename = [filename]

    filepaths = []
    for fn in filename:
        fs = fn.split('_')

        year = fs[-2][:4]
        month = fs[-2][4:6]
        filepaths.append(
                './mms/' + '/'.join(fs[:-2]) + f'/{year}/{month}/{fn}')

    if len(filepaths) == 1:
        return filepaths[0]
    return filepaths



_MMS_DATA_BASE_URL = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/'

def get_file_list(start_date,end_date,data_rate = 'fast', datalevel = 'l2',
                  datatype = 'dis-dist', instrument = 'fpi'):
    """
    Get a list of files from the MMS Science Data center
    """
    url = f'{_MMS_DATA_BASE_URL}file_info/science?'
    url += f'start_date={start_date}&end_date={end_date}&sc_id=mms1'
    url += f'&instrument_id={instrument}'
    url += f'&data_rate_mode={data_rate}&data_level={datalevel}'
    if not datatype is None:
        url += f'&descriptor={datatype}'

    with requests.Session() as session:
        r = session.get(url)
        files = r.json()['files']
        r.close()

    return files

def download_cdf_files(rootdir, cdf_filepaths, session = None):
    """
    Download CDF files from the MMS Science Data Center based on a filelist.
    """
    close_session = False
    if session is None:
        close_session = True
        session = requests.Session()

    t_cnt = len(cdf_filepaths)
    rootdir = rootdir[:-1] if rootdir[-1] == '/' else rootdir

    for cnt, filepath in enumerate(cdf_filepaths,1):
        dirpath, filename = path.split(filepath)
        makedirs(f'{rootdir}/{dirpath}', exist_ok=True)
        url_file = f'{_MMS_DATA_BASE_URL}download/science?file={filename}'
        filepath = path.abspath(f'{rootdir}/{dirpath}/{filename}')
        if path.isfile(filepath):
            print(f'({cnt}/{t_cnt}): File {filename} exists, skipping.')
        else:
            print(f'({cnt}/{t_cnt}): Downloading file {filename}')
            download_file_with_status(url_file, filepath, session)
    if close_session:
        session.close()
