import pytest
import os, sys
from cellpose import utils
from urllib.request import urlopen
from urllib.parse import urlparse

from pathlib import Path

@pytest.fixture()
def image_names():
    image_names = ['gray_2D.png', 'rgb_2D.png', 'rgb_2D_tif.tif', 'gray_3D.tif', 'rgb_3D.tif']
    return image_names

@pytest.fixture()
def data_dir(image_names):
    cp_dir = Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    data_dir = cp_dir.joinpath('data')
    data_dir.mkdir(exist_ok=True)
    data_dir_2D = data_dir.joinpath('2D')
    data_dir_2D.mkdir(exist_ok=True)
    data_dir_3D = data_dir.joinpath('3D')
    data_dir_3D.mkdir(exist_ok=True)

    for image_name in image_names:
        url = 'http://www.cellpose.org/static/data/' + image_name
        if '2D' in image_name:
            cached_file = str(data_dir_2D.joinpath(image_name))
            ext = '.png'
        else:
            cached_file = str(data_dir_3D.joinpath(image_name))
            ext = '.tif'
        if not os.path.exists(cached_file):
            utils.download_url_to_file(url, cached_file)
        
        # check if mask downloaded (and clear potential previous test data)
        name = os.path.splitext(cached_file)[0]
        mask_file = name + '_cp_masks' + ext
        if os.path.exists(mask_file):
            os.remove(mask_file)
        cached_mask_files = [name + '_cyto_masks' + ext, name + '_nuclei_masks' + ext]
        for cached_mask_file in cached_mask_files:
            url = 'http://www.cellpose.org/static/data/' + os.path.split(cached_mask_file)[-1]
            if not os.path.exists(cached_mask_file):
                print(cached_mask_file)
                utils.download_url_to_file(url, cached_mask_file, progress=True)
    return data_dir