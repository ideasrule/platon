from __future__ import print_function
from pkg_resources import resource_filename

import requests
import sys
import urllib2
import zipfile
import os
import shutil

def get_data():
    url = "https://github.com/ideasrule/platon/archive/beta.zip"

    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading {0}: {1} MB".format(file_name, file_size/2**20))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        percentage = int(100 * file_size_dl / file_size)
        status = "{} MB  [{}%]".format(file_size_dl/2**20, percentage)
        print(status, end="\r")

    f.close()

    zip_ref = zipfile.ZipFile("beta.zip", 'r')
    zip_ref.extractall(".")
    zip_ref.close()

    shutil.move("platon-beta/platon/data", resource_filename(__name__, "data"))
    shutil.rmtree("platon-beta")
    os.remove("beta.zip")

#get_data()
