import pytest
import os, sys, shutil
from cellpose import utils
import zipfile

from pathlib import Path


@pytest.fixture()
def image_names():
    image_names = [ f"{str(i).zfill(3)}_img.tif" for i in range(0, 91) ] 
    return image_names

def extract_zip(cached_file, url, data_path):
    utils.download_url_to_file(url, cached_file)        
    with zipfile.ZipFile(cached_file,"r") as zip_ref:
        zip_ref.extractall(data_path)

@pytest.fixture()
def data_dir(image_names):
    cp_dir = Path.home().joinpath(".cellpose")
    cp_dir.mkdir(exist_ok=True)
    extract_zip(cp_dir.joinpath("data.zip"), "https://osf.io/download/67f022eb033d25194f82a4ee/", cp_dir)
    data_dir = cp_dir.joinpath("data")
    return data_dir
    