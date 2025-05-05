def test_cellpose_imports_without_error():
    import cellpose
    from cellpose import models, core
    model = models.CellposeModel()


def test_model_zoo_imports_without_error():
    from cellpose import models, denoise
    for model_name in models.MODEL_NAMES:
        model = models.CellposeModel(pretrained_model=model_name)

def test_gui_imports_without_error():
    from cellpose import gui


def test_gpu_check():
    from cellpose import core
    core.use_gpu()


def itest_model_dir():
    import os, pathlib
    import numpy as np
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(
        pathlib.Path.home().joinpath('.cellpose'))

    from cellpose import models
    model = models.CellposeModel(pretrained_model='cpsam')
    masks = model.eval(np.random.randn(256, 256))[0]
    assert masks.shape == (256, 256)
