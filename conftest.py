import pytest
import os, sys, shutil
from cellpose import utils

from pathlib import Path

@pytest.fixture()
def image_names():
    image_names = ['gray_2D.png', 'rgb_2D.png', 'rgb_2D_tif.tif', 'gray_3D.tif', 'rgb_3D.tif',
                    'segment_80x224x448_input.tiff', 'segment_80x224x448_expected.tiff']
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
    data_dir_distributed = data_dir.joinpath('distributed')
    data_dir_distributed.mkdir(exist_ok=True)

    for i,image_name in enumerate(image_names):
        url = 'http://www.cellpose.org/static/data/' + image_name
        if i<3:
            cached_file = str(data_dir_2D.joinpath(image_name))
            ext = '.png'
        elif i<5:
            cached_file = str(data_dir_3D.joinpath(image_name))
            ext = '.tif'
        else:
            cached_file = str(data_dir_distributed.joinpath(image_name))
            ext = '.tif'
        if not os.path.exists(cached_file):
            utils.download_url_to_file(url, cached_file)
        
        # check if mask downloaded (and clear potential previous test data)
        if i<2:
            train_dir = data_dir_2D.joinpath('train')
            train_dir.mkdir(exist_ok=True)
            shutil.copyfile(cached_file, train_dir.joinpath(image_name))

        if i<5:
            name = os.path.splitext(cached_file)[0]
            mask_file = name + '_cp_masks' + ext
            if os.path.exists(mask_file):
                os.remove(mask_file)
            cached_mask_files = [name + '_cyto_masks' + ext, name + '_nuclei_masks' + ext]
            for c,cached_mask_file in enumerate(cached_mask_files):
                url = 'http://www.cellpose.org/static/data/' + os.path.split(cached_mask_file)[-1]
                if not os.path.exists(cached_mask_file):
                    print(cached_mask_file)
                    utils.download_url_to_file(url, cached_mask_file, progress=True)
                if i<2 and c==0:
                    shutil.copyfile(cached_mask_file, 
                        train_dir.joinpath(os.path.splitext(image_name)[0] + '_cyto_masks' + ext))       
    return data_dir
