from urllib.request import urlopen
from pkg_resources import resource_filename
from platon import __data_url__, __gdrive__, __md5sum__

import sys
import zipfile
import os
import hashlib
import ssl

# Flag to enable or disable Google Drive as the CDN
GDRIVE_CDN = False  # Set to False to use the default __data_url__

# Try to import gdown only if GDRIVE_CDN is True
if GDRIVE_CDN:
    try:
        import gdown
    except ImportError:
        print("Error: gdown is not installed. run 'pip install gdown'")
        sys.exit(1)  # Exit the program if gdown is not install


def get_data_if_needed():
    if not os.path.isdir(resource_filename(__name__, "data/")):
        get_data(resource_filename(__name__, "./"))
        
    with open(resource_filename(__name__, "md5sum")) as f:
        curr_md5sum = f.read().strip()

    if __md5sum__ != curr_md5sum:
        print("Warning: data files are out of date. To update, remove the PLATON data directory ({}) and PLATON will automatically download the latest data files on the next run.".format(resource_filename(__name__, "data/")))


def get_data(target_dir):
    # Determine which CDN to use
    use_gdrive = GDRIVE_CDN and bool(__gdrive__)  # Use GDrive if True
    url = __gdrive__ if use_gdrive else __data_url__
    filename = "data_gdrive.zip" if use_gdrive else "data.zip"
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Skipping.")
    else:
        if use_gdrive:
            print(f"Downloading from Google Drive: {url}")
            gdown.download(url, filename, quiet=False)  # Use gdown for Google Drive downloads
        else:
            MB_TO_BYTES = 2**20
            print("Using URL:", url)

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            u = urlopen(url, context=ctx)
            with open(filename, 'wb') as f:
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
            print("\nDownload finished!")

    print("Extracting...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction finished!")

    with open(filename, "rb") as f:
        curr_md5sum = hashlib.md5(f.read()).hexdigest()

    if curr_md5sum != __md5sum__:
        raise RuntimeError("Downloaded data file is corrupt (wrong md5sum). Please try again.")

    with open(resource_filename(__name__, "md5sum"), "w") as f:
        f.write(curr_md5sum)

    os.remove(filename)
