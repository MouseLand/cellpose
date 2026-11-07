"""Test figure handling without loading paper datasets or pretrained models."""

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def real_examples():
    # Loading only this function avoids the paper script's optional training
    # packages and import-time logger setup. Plotting and NumPy output are real.
    source = Path(__file__).resolve().parents[1] / "paper" / "3.0" / "analysis.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "real_examples")

    class Segmenter:
        def __init__(self, **kwargs):
            pass

        def eval(self, images, **kwargs):
            if isinstance(images, list):
                masks = [np.ones(image.shape[-2:], dtype=np.uint16) for image in images]
            else:
                masks = np.ones(images.shape[-2:], dtype=np.uint16)
            return masks, None, None

    class Denoiser:
        def __init__(self, **kwargs):
            pass

        def eval(self, images, **kwargs):
            return [image + 0.25 for image in images]

    def read_image(path):
        if path.name == "stack.tif":
            return np.arange(4 * 16 * 4 * 5, dtype=float).reshape(4, 16, 4, 5)
        return np.arange(20, dtype=float).reshape(4, 5)

    def average_precision(actual, predicted, threshold):
        ap = np.tile(np.linspace(0.2, 0.9, len(threshold)), (len(actual), 1))
        return ap, None, None, None

    namespace = {
        "Path": Path, "np": np, "plt": plt, "natsorted": sorted,
        "io": SimpleNamespace(imread=read_image),
        "transforms": SimpleNamespace(normalize99=lambda image: image.copy()),
        "models": SimpleNamespace(CellposeModel=Segmenter),
        "denoise": SimpleNamespace(DenoiseModel=Denoiser),
        "metrics": SimpleNamespace(average_precision=average_precision),
        "noise2self": SimpleNamespace(train_per_image=lambda image: image.copy()),
        "noise2void": SimpleNamespace(train_per_image=lambda image: image.copy()),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    initial_figures = set(plt.get_fignums())
    try:
        yield namespace["real_examples"]
    finally:
        for number in set(plt.get_fignums()) - initial_figures:
            plt.close(number)


@pytest.fixture
def example_data(tmp_path):
    for name in ("Projection_Flywing", "Denoising_Tribolium"):
        directory = tmp_path / name / "test_data"
        directory.mkdir(parents=True)
        filenames = [f"proj_C{i}.tif" for i in range(4)] if name == "Projection_Flywing" else ["stack.tif"]
        for filename in filenames:
            (directory / filename).touch()
    return tmp_path


def read_outputs(folder):
    return {
        str(path.relative_to(folder)): np.load(path, allow_pickle=True).item()
        for path in sorted(folder.glob("*/test_data/*_masks.npy"))
    }


def assert_same_outputs(left, right):
    assert left.keys() == right.keys()
    for path in left:
        assert left[path].keys() == right[path].keys()
        for key in left[path]:
            np.testing.assert_array_equal(left[path][key], right[path][key])


def test_default_behavior_still_shows_plots(real_examples, example_data, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    real_examples(example_data)
    assert len(shown) == 2
    assert all(len(figure.axes[0].lines) == 6 for figure in shown)
    assert not list(example_data.glob("*/test_data/cp_ap.png"))
    assert len(read_outputs(example_data)) == 6


@pytest.mark.parametrize("as_string", [False, True])
def test_save_figures_is_noninteractive(real_examples, example_data, monkeypatch, as_string):
    initial_figures = set(plt.get_fignums())
    monkeypatch.setattr(plt, "show", lambda: None)
    real_examples(example_data)
    expected = read_outputs(example_data)
    for number in set(plt.get_fignums()) - initial_figures:
        plt.close(number)
    sentinel = plt.figure()

    def fail_show():
        pytest.fail("Saving figures must not open an interactive display")

    monkeypatch.setattr(plt, "show", fail_show)
    folder = str(example_data) if as_string else example_data
    real_examples(folder, save_figures=True)
    outputs = sorted(example_data.glob("*/test_data/cp_ap.png"))
    assert len(outputs) == 2
    assert {path.parents[1].name for path in outputs} == {
        "Projection_Flywing", "Denoising_Tribolium"
    }
    for path in outputs:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        pixels = mpimg.imread(path)
        assert pixels.ndim == 3 and pixels.shape[0] > 0 and pixels.shape[1] > 0
    assert set(plt.get_fignums()) == initial_figures | {sentinel.number}
    assert_same_outputs(expected, read_outputs(example_data))


def test_save_failure_closes_only_its_figure(real_examples, example_data, monkeypatch):
    from matplotlib.figure import Figure

    initial_figures = set(plt.get_fignums())
    sentinel = plt.figure()

    def fail_save(self, *args, **kwargs):
        raise OSError("synthetic output failure")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="synthetic output failure"):
        real_examples(example_data, save_figures=True)
    assert set(plt.get_fignums()) == initial_figures | {sentinel.number}
