import time
import numpy as np
import pytest
from cellpose import utils, models, vit_sam
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
    """ This is functionally identical to CellposeModel but uses mock class """
    use_gpu = torch.cuda.is_available()
    use_mps = 'mps' if torch.backends.mps.is_available() else False
    gpu = use_gpu or use_mps
    model = MockCellposeModel(24, gpu=gpu)
    yield model


@pytest.fixture()
def cellposemodel_fixture_2layer():
    """ This is only uses 2 transformer blocks for speed """
    use_gpu = torch.cuda.is_available()
    use_mps = 'mps' if torch.backends.mps.is_available() else False
    gpu = use_gpu or use_mps
    model = MockCellposeModel(n_keep_layers=2, gpu=gpu)
    yield model


class MockTransformer(vit_sam.Transformer):
    def __init__(self, use_layers: int):
        """ use_layers: the number of layers use starting from the first layer """
        super().__init__()

        self.use_layers = use_layers
        self.layer_idxs = np.linspace(0, 23, self.use_layers, dtype=int)

    def forward(self, x):
        # same progression as SAM until readout
        x = self.encoder.patch_embed(x)
        
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        
        # only use self.use_layers layers
        for layer_idx in self.layer_idxs:
            x = self.encoder.blocks[layer_idx](x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        # readout is changed here
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride = self.ps, padding = 0)
        
        # maintain the second output of feature size 256 for backwards compatibility
        return x1, torch.randn((x.shape[0], 256), device=x.device)


class MockCellposeModel(models.CellposeModel):
    def __init__(self, n_keep_layers=2, gpu=False):
        super().__init__(gpu=gpu)

        self.net = MockTransformer(n_keep_layers)
        self.net.to(self.device)
        self.net.load_model(Path().home() / '.cellpose/models/cpsam', device=self.device)

    def eval(self, *args, **kwargs):
        tic = time.time()
        res = super().eval(*args, **kwargs)
        toc = time.time()

        print(f'eval() time elapsed: {toc-tic}')
        return res
