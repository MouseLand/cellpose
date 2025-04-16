import pytest
import os, sys, shutil
from cellpose import utils
import zipfile

from pathlib import Path


@pytest.fixture()
def image_names():
    image_names = ['rgb_2D_tif.tif', 'rgb_2D.png', 'gray_2D.png']
    return image_names

@pytest.fixture()
def image_names_3d():
    image_names_3d = ['rgb_3D.tif', 'gray_3D.tif']
    return image_names_3d

def extract_zip(cached_file, url, data_path):
    utils.download_url_to_file(url, cached_file)        
    with zipfile.ZipFile(cached_file,"r") as zip_ref:
        zip_ref.extractall(data_path)

@pytest.fixture()
def data_dir(image_names):
    cp_dir = Path.home().joinpath(".cellpose")
    cp_dir.mkdir(exist_ok=True)
    # TODO: setup the remote data later, for now use local data
    # extract_zip(cp_dir.joinpath("data.zip"), "https://osf.io/download/67f022eb033d25194f82a4ee/", cp_dir)
    data_dir = cp_dir.joinpath("data")
    return data_dir
    