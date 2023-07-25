def test_cellpose_imports_without_error():
    import cellpose
    from cellpose import models, core
    model = models.CellposeModel()
    model = core.UnetModel()

def test_model_zoo_imports_without_error():
    from cellpose import models 
    for model_name in models.MODEL_NAMES:
        model = models.CellposeModel(model_type=model_name)

def test_gui_imports_without_error():
    from cellpose import gui

def test_gpu_check():
#     from cellpose import models
#     models.use_gpu()
    from cellpose import core
    core.use_gpu()


def test_model_dir():
    import os, pathlib
    import numpy as np
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(pathlib.Path.home().joinpath('.cellpose'))

    from cellpose import models
    model = models.CellposeModel(net_avg=False, pretrained_model='cyto')
    masks = model.eval(np.random.randn(224,224))[0]
    assert masks.shape==(224,224)