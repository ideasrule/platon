from __future__ import print_function
from future.moves.urllib.request import urlopen

from pkg_resources import resource_filename
from platon import __data_url__

import sys
import zipfile
import os
import shutil

def get_data(target_dir):
    MB_TO_BYTES = 2**20
    filename = "data.zip"
    u = urlopen(__data_url__)
    f = open(filename, 'wb')
    try:
        # Python 3
        file_size = int(u.getheader("Content-Length"))
    except AttributeError:
        # Python 2
        file_size = int(u.info().getheaders("Content-Length")[0])

    print("Downloading {}: {:.0f} MB".format(filename, file_size/MB_TO_BYTES))

    bytes_downloaded = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        bytes_downloaded += len(buffer)
        f.write(buffer)
        percentage = int(100 * bytes_downloaded / file_size)
        status = "{:.0f} MB  [{}%]".format(bytes_downloaded/MB_TO_BYTES, percentage)
        print(status, end="\r")

    f.close()

    print("\nExtracting...")
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()

    print("Extraction finished!")
    os.remove(filename)


