"""
Utils for file downloads.
"""

import functools
from os import path
from shutil import copyfileobj, copy
from tempfile import NamedTemporaryFile
from tqdm.auto import tqdm
import requests

def missing_files(files,filepath=''):
    """
    Check for missing files.
    """
    missing = []
    for filename in files:
        if not path.isfile(path.abspath(f'{filepath}{filename}')):
            missing.append(filename)

    if len(missing) == 0:
        return None

    return missing

def download_file_with_status(url_file, filepath, session = None):
    """
    Download files with a progress bar.
    """
    close_session = False
    if session is None:
        close_session = True
        session = requests.Session()

    r = session.get(url_file, stream=True, verify=True)

    # Code for status bar from:
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url_file} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with NamedTemporaryFile() as ftmp:
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
            with open(ftmp.name, 'wb') as f:
                copyfileobj(r_raw, f)

        # If the download was successful, copy to data directory.
        # This prevents a failed (or aborted) download to block
        # future downloads.
        copy(ftmp.name, filepath)
        r.close()

    if close_session:
        session.close()
