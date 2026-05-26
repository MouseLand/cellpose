import time
import numpy as np
import pytest
from cellpose import utils, models
import zipfile
import torch
import torch.nn.functional as F
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture()
def image_names():
    image_names = ['rgb_2D_tif.tif', 'rgb_2D.png', 'gray_2D.png']
    return image_names


@pytest.fixture()
def image_names_3d():
    image_names_3d = ['rgb_3D.tif', 'gray_3D.tif']
    return image_names_3d


def extract_zip(cached_file, url, data_path):
    if not cached_file.exists():
        utils.download_url_to_file(url, cached_file)        
        with zipfile.ZipFile(cached_file,"r") as zip_ref:
            zip_ref.extractall(data_path)

@pytest.fixture()
def data_dir(image_names):
    cp_dir = Path.home().joinpath(".cellpose")
    cp_dir.mkdir(exist_ok=True)
    extract_zip(cp_dir.joinpath("data.zip"), "https://osf.io/download/s52q3/", cp_dir)
    data_dir = cp_dir.joinpath("data")
    return data_dir

    
@pytest.fixture()
def cellposemodel_fixture_24layer():
    """ Load full transformer model """
    use_gpu = torch.cuda.is_available()
    use_mps = 'mps' if torch.backends.mps.is_available() else False
    gpu = use_gpu or use_mps
    model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
    yield model


@pytest.fixture()
def cellposemodel_fixture_2layer():
    """ This only uses 2 transformer blocks and vitb for speed """
    use_gpu = torch.cuda.is_available()
    use_mps = 'mps' if torch.backends.mps.is_available() else False
    gpu = use_gpu or use_mps
    model = models.CellposeModel(gpu=gpu, pretrained_model="cpdino-vitb")
    model.net.encoder.blocks = model.net.encoder.blocks[:2]
    yield model
