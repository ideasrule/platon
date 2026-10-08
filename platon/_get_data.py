from urllib.request import urlopen

from platon import __data_url__, __md5sum__

import zipfile
import os
import hashlib
import tempfile
from pathlib import Path, PurePosixPath

def get_data_if_needed():
    basedir = Path(__file__).resolve().parent
    if not os.path.isdir(basedir / "data"):
        get_data(basedir)
        
    with open(str(basedir / "md5sum")) as f:
        curr_md5sum = f.read().strip()

    if __md5sum__ != curr_md5sum:
        print("Warning: data files are out of date.  To update, remove the PLATON data directory ({}) and PLATON will automatically download the latest data files on the next run.".format(basedir / "data"))
        

def get_data(target_dir):
    """Download and verify the archive before installing its data directory.

    Staging is on the destination filesystem so the final directory rename
    is atomic. Failed downloads or extraction leave no partial installation.
    """
    MB_TO_BYTES = 2**20
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / "data"
    if destination.exists():
        raise FileExistsError("Data directory already exists: {}".format(destination))
    print("Data URL", __data_url__)

    with tempfile.TemporaryDirectory(prefix=".platon-download-",
                                     dir=target_dir) as staging_dir:
        staging = Path(staging_dir)
        filename = staging / "data.zip"
        checksum = hashlib.md5()
        # urlopen's default HTTPS context verifies certificates and hostnames.
        with urlopen(__data_url__) as response, filename.open("wb") as output:
            length = response.getheader("Content-Length")
            file_size = int(length) if length is not None else None
            bytes_downloaded = 0
            while True:
                block = response.read(2**20)
                if not block:
                    break
                output.write(block)
                checksum.update(block)
                bytes_downloaded += len(block)
                status = "{:.0f} MB".format(bytes_downloaded / MB_TO_BYTES)
                if file_size:
                    status += "  [{}%]".format(int(100 * bytes_downloaded / file_size))
                print(status, end="\r")

        curr_md5sum = checksum.hexdigest()
        if curr_md5sum != __md5sum__:
            raise RuntimeError(
                "Downloaded data file is corrupt (wrong md5sum). Please try again.")

        print("\nExtracting...")
        with zipfile.ZipFile(filename) as archive:
            # The archive may only populate data/, never package source files
            # or paths outside the staging directory.
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or \
                   "\\" in member.filename or not path.parts or path.parts[0] != "data":
                    raise ValueError("Invalid data archive path: {}".format(member.filename))
            archive.extractall(staging)

        if not (staging / "data").is_dir():
            raise ValueError("Downloaded archive does not contain a data directory")
        checksum_path = staging / "md5sum"
        checksum_path.write_text(curr_md5sum)
        # Install the checksum first: once data/ becomes visible, its checksum
        # is already present. A failed rename leaves data/ absent and retryable.
        os.replace(checksum_path, target_dir / "md5sum")
        os.replace(staging / "data", destination)
    print("Extraction finished!")
