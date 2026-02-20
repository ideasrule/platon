from urllib.request import urlopen

import importlib.resources
from platon import __data_url__, __md5sum__

import sys
import zipfile
import os
import hashlib
import ssl
from pathlib import Path

def get_data_if_needed():
    basedir = Path(__file__).resolve().parent
    if not os.path.isdir(basedir / "data"):
        get_data(basedir)
        
    with open(str(basedir / "md5sum")) as f:
        curr_md5sum = f.read().strip()

    if __md5sum__ != curr_md5sum:
        print("Warning: data files are out of date.  To update, remove the PLATON data directory ({}) and PLATON will automatically download the latest data files on the next run.".format(basedir / "data"))
        

def get_data(target_dir):
    MB_TO_BYTES = 2**20
    filename = "data.zip"
    print("Data URL", __data_url__)

    #Bad! Dangerous! But necessary...get a real certificate, Caltech!
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    u = urlopen(__data_url__, context=ctx)
    f = open(filename, 'wb')

    #Only for Python 3, because we don't support Python 2 anymore
    file_size = int(u.getheader("Content-Length"))

    print("Downloading {}: {:.0f} MB".format(
        filename, file_size / MB_TO_BYTES))

    bytes_downloaded = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        bytes_downloaded += len(buffer)
        f.write(buffer)
        percentage = int(100 * bytes_downloaded / file_size)
        status = "{:.0f} MB  [{}%]".format(
            bytes_downloaded / MB_TO_BYTES, percentage)
        print(status, end="\r")

    f.close()

    print("\nExtracting...")
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()

    print("Extraction finished!")
    with open(filename, "rb") as f:
        curr_md5sum = hashlib.md5(f.read()).hexdigest()

    if curr_md5sum != __md5sum__:
        raise RuntimeError("Downloaded data file is corrupt (wrong md5sum).  Please try again.")

    basedir = Path(__file__).resolve().parent
    with open(basedir / "md5sum", "w") as f:
        f.write(curr_md5sum)
        
    os.remove(filename)
