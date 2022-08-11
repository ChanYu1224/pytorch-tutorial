import os
import subprocess
import zipfile
from __future__ import unicode_literals, print_function, division
import glob
from io import open

subprocess.run("wget https://download.pytorch.org/tutorial/data.zip", shell=True, check=True)
with zipfile.ZipFile("./data.zip") as zipfile:
    zipfile.extractall(".")

def find_files(path):
    return glob.glob(path)


