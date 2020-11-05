
def test_cellpose_imports_without_error():
    import cellpose
    from cellpose import models
    model = models.CellposeModel()
    model = models.UnetModel()

def test_gui_imports_without_error():
    from cellpose import gui

def test_gpu_check():
    from cellpose import models
    models.use_gpu()