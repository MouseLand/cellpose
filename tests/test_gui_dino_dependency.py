"""GUI model selection must check DINO before touching the model cache."""
import builtins
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")
pytest.importorskip("pyqtgraph")

from cellpose import models
from cellpose.gui.gui import MainW


@pytest.fixture
def gui_model_selection(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    monkeypatch.setattr(models, "MODEL_DIR", model_dir)
    download = Mock(side_effect=AssertionError("unexpected weight download"))
    monkeypatch.setattr(models.utils, "download_url_to_file", download)
    window = SimpleNamespace(ModelChooseB=Mock(), ModelChooseC=Mock())
    return window, model_dir, download


@pytest.fixture
def missing_dino(monkeypatch):
    original_import = builtins.__import__
    failure = ModuleNotFoundError("No module named 'dinov3'", name="dinov3")
    attempts = []

    def import_without_dino(name, *args, **kwargs):
        if name == "dinov3" or name.startswith("dinov3."):
            attempts.append(name)
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dino)
    return failure, attempts


@pytest.mark.parametrize("name", ["cpdino", "cpdino-vitb"])
def test_gui_dino_missing_before_cache(gui_model_selection, missing_dino, name):
    window, model_dir, download = gui_model_selection
    failure, attempts = missing_dino
    window.ModelChooseB.currentText.return_value = name

    with pytest.raises(ImportError) as caught:
        MainW.get_model_path(window)

    assert "CPDINO" in str(caught.value)
    assert "pip install git+https://github.com/facebookresearch/dinov3" in str(caught.value)
    assert caught.value.__cause__ is failure
    assert attempts
    download.assert_not_called()
    assert not model_dir.exists()


@pytest.mark.parametrize("name", ["cpsam", "cpsam_v2"])
def test_gui_cpsam_cache_without_dino(gui_model_selection, missing_dino, name):
    window, model_dir, download = gui_model_selection
    _, attempts = missing_dino
    window.ModelChooseB.currentText.return_value = name

    def provide_checkpoint(url, path, progress):
        assert url.endswith("/" + name)
        assert progress is True
        # Path lookup does not need a real checkpoint.
        model_dir.joinpath(name).touch()

    download.side_effect = provide_checkpoint
    MainW.get_model_path(window)
    MainW.get_model_path(window)

    assert window.current_model == name
    assert window.current_model_path == os.fspath(model_dir / name)
    download.assert_called_once()
    assert attempts == []


@pytest.mark.parametrize("name", ["cpdino", "cpdino-vitb"])
def test_gui_installed_dino_cache(gui_model_selection, name):
    pytest.importorskip("dinov3")
    window, model_dir, download = gui_model_selection
    window.ModelChooseB.currentText.return_value = name

    def provide_checkpoint(url, path, progress):
        assert url.endswith("/" + name)
        assert progress is True
        # Only resolving paths here; model loading is covered separately.
        model_dir.joinpath(name).touch()

    download.side_effect = provide_checkpoint
    MainW.get_model_path(window)
    MainW.get_model_path(window)

    assert window.current_model == name
    assert window.current_model_path == os.fspath(model_dir / name)
    download.assert_called_once()


@pytest.mark.parametrize("name", ["cpdino", "cpdino-vitb"])
def test_gui_custom_dino_name_keeps_checkpoint_error(
        gui_model_selection, missing_dino, name):
    window, model_dir, download = gui_model_selection
    _, attempts = missing_dino
    model_dir.mkdir()
    path = model_dir / name
    # A valid archive with a malformed DINO state dict has its own error.
    models.torch.save({"encoder.cls_token": models.torch.zeros(1)}, path)
    window.ModelChooseC.currentText.return_value = name

    MainW.get_model_path(window, custom=True)

    assert window.current_model_path == os.fspath(path)
    with pytest.raises(KeyError, match="encoder.patch_embed.proj.weight"):
        models.CellposeModel(pretrained_model=window.current_model_path,
                             device=models.torch.device("cpu"))
    download.assert_not_called()
    assert attempts == []
