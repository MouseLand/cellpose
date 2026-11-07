"""Optional DINO imports, isolated from pytest's imports and process state."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PRELUDE = """
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
root = Path.cwd()
work = Path(sys.argv[1])
command = 'pip install git+https://github.com/facebookresearch/dinov3'

class ImportFailure(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, failure):
        self.failure = failure
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dinov3' or fullname.startswith('dinov3.'):
            self.attempts.append(fullname)
        if self.failure == 'nested':
            if fullname == 'dinov3.hub.utils':
                return importlib.util.spec_from_loader(fullname, self)
        elif fullname == 'dinov3':
            raise self.failure

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        try:
            import cellpose_test_missing_dino_dependency
        except ModuleNotFoundError as exc:
            raise ImportError('DINO nested dependency failed') from exc

def install_failure(kind):
    failure = (ModuleNotFoundError("No module named 'dinov3'", name='dinov3')
               if kind == 'missing' else kind)
    finder = ImportFailure(failure)
    sys.meta_path.insert(0, finder)
    return finder

def check_error(error, kind):
    assert command in str(error)
    assert 'CPDINO' in str(error)
    cause = error.__cause__
    if kind == 'missing':
        assert isinstance(cause, ModuleNotFoundError)
        assert cause.name == 'dinov3'
    else:
        assert str(cause) == 'DINO nested dependency failed'
        assert isinstance(cause.__cause__, ModuleNotFoundError)
        assert cause.__cause__.name == 'cellpose_test_missing_dino_dependency'

def checkpoint(path, dim):
    torch.save({'encoder.cls_token': torch.zeros(1),
                'encoder.patch_embed.proj.weight': torch.zeros(dim, 1, 1, 1)}, path)

# Only the resource-heavy transformer constructor is replaced. The installed
# DINO module, both factory functions, and Cellpose constructors remain real.
class TinyDino(nn.Module):
    configs = []

    def __init__(self, **kwargs):
        super().__init__()
        self.configs.append(kwargs)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, kwargs['embed_dim'], 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, kwargs['embed_dim']))

    def init_weights(self):
        pass
"""


@pytest.fixture
def run_python(tmp_path):
    def run(source):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root),
                   CELLPOSE_LOCAL_MODELS_PATH=str(tmp_path / "models"),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                   VECLIB_MAXIMUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        # Process isolation discards module/import hooks, RNG and torch global
        # flags (including vit.py's TF32 setting), even when assertions fail.
        result = subprocess.run(
            [sys.executable, "-c", PRELUDE + textwrap.dedent(source), str(tmp_path)],
            cwd=root, env=env, text=True, capture_output=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return run


@pytest.mark.parametrize("kind", ["missing", "nested"])
@pytest.mark.parametrize("entry", ["direct", "builtin", "local"])
def test_dino_import_error(run_python, kind, entry):
    if kind == "nested" and importlib.util.find_spec("dinov3") is None:
        pytest.skip("nested import control requires the optional DINO package")
    run_python(f"""
        finder = install_failure({kind!r})
        from cellpose import models, vit
        assert Path(models.__file__).resolve() == root / 'cellpose/models.py'
        assert Path(vit.__file__).resolve() == root / 'cellpose/vit.py'
        download = Mock(side_effect=AssertionError('unexpected weight download'))
        with patch.object(models.utils, 'download_url_to_file', download):
            for name, variant, dim in [('cpdino', 'vitl', 1024),
                                        ('cpdino-vitb', 'vitb', 768)]:
                if {entry!r} == 'local':
                    path = work / name
                    checkpoint(path, dim)
                    assert models.get_backbone(path) == 'dino_' + variant
                with pytest.raises(ImportError) as caught:
                    if {entry!r} == 'direct':
                        vit.CPDINO(model_name=variant)
                    else:
                        models.CellposeModel(
                            pretrained_model=str(path) if {entry!r} == 'local' else name,
                            device=torch.device('cpu'))
                check_error(caught.value, {kind!r})
            download.assert_not_called()
        assert finder.attempts
        assert not models.MODEL_DIR.exists()
    """)


@pytest.mark.parametrize("exception", ["RuntimeError", "KeyboardInterrupt", "SystemExit"])
def test_non_import_errors_propagate(run_python, exception):
    run_python(f"""
        original = {exception}('backend failure')
        install_failure(original)
        from cellpose import models, vit
        download = Mock(side_effect=AssertionError('unexpected weight download'))
        with patch.object(models.utils, 'download_url_to_file', download):
            for variant, name in [('vitl', 'cpdino'), ('vitb', 'cpdino-vitb')]:
                with pytest.raises({exception}) as caught:
                    vit.CPDINO(model_name=variant)
                assert caught.value is original
                with pytest.raises({exception}) as caught:
                    models.CellposeModel(pretrained_model=name, device=torch.device('cpu'))
                assert caught.value is original
        download.assert_not_called()
        assert not models.MODEL_DIR.exists()
    """)


@pytest.mark.parametrize("variant,name,dim", [("vitl", "cpdino", 1024),
                                             ("vitb", "cpdino-vitb", 768)])
def test_installed_dino_factories(run_python, variant, name, dim):
    if importlib.util.find_spec("dinov3") is None:
        pytest.skip("healthy import control requires the optional DINO package")
    run_python(f"""
        from dinov3.models import vision_transformer
        from cellpose import models, vit
        download = Mock(side_effect=AssertionError('unexpected DINO weight download'))
        with patch.object(vision_transformer, 'DinoVisionTransformer', TinyDino), \
                patch.object(torch.hub, 'load_state_dict_from_url', download):
            net = vit.CPDINO(model_name={variant!r})
            assert TinyDino.configs[-1]['embed_dim'] == {dim}
            assert net.encoder.patch_embed.proj.stride == (8, 8)
            assert net.encoder.patch_embed.proj.padding == (4, 4)
            local = work / 'custom-model'
            net.save_model(local)
            def provide_checkpoint(url, path, progress):
                assert url.endswith('/' + {name!r})
                torch.save(net.state_dict(), path)
            with patch.object(models.utils, 'download_url_to_file',
                              side_effect=provide_checkpoint) as fetch:
                for selected in [{name!r}, {name!r}, str(local)]:
                    model = models.CellposeModel(pretrained_model=selected,
                        device=torch.device('cpu'), use_bfloat16=False)
                    assert model.backbone == 'dino_' + {variant!r}
                    assert isinstance(model.net, vit.CPDINO)
                    assert isinstance(model.net.encoder, TinyDino)
                    for key, value in net.state_dict().items():
                        assert torch.equal(model.net.state_dict()[key], value)
                assert fetch.call_count == 1
            assert len(TinyDino.configs) == 4
        download.assert_not_called()
    """)


def test_retry_after_failed_import(run_python):
    if importlib.util.find_spec("dinov3") is None:
        pytest.skip("retry control requires the optional DINO package")
    run_python("""
        finder = install_failure('missing')
        from cellpose import vit
        with pytest.raises(ImportError):
            vit.CPDINO()
        sys.meta_path.remove(finder)
        from dinov3.models import vision_transformer
        with patch.object(vision_transformer, 'DinoVisionTransformer', TinyDino), \
                patch.object(torch.hub, 'load_state_dict_from_url',
                             side_effect=AssertionError('unexpected weight download')):
            for variant in ['vitl', 'vitb']:
                assert isinstance(vit.CPDINO(model_name=variant).encoder, TinyDino)
    """)


def test_cpsam_and_utilities_do_not_import_dino(run_python):
    run_python("""
        finder = install_failure('missing')
        from cellpose import models, vit, utils, transforms

        # Explicit small SAM factory double; exercise the actual CPSAM code.
        def tiny_sam(checkpoint):
            encoder = nn.Module()
            encoder.patch_embed = nn.Module()
            encoder.patch_embed.proj = nn.Conv2d(3, 4, 16)
            encoder.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, 4))
            encoder.blocks = nn.ModuleList()
            encoder.neck = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Identity())
            sam = nn.Module()
            sam.image_encoder = encoder
            return sam

        with patch.dict(vit.sam_model_registry, {'vit_l': tiny_sam}):
            net = vit.CPSAM()
            local = work / 'custom-sam'
            net.save_model(local)
            def provide_checkpoint(url, path, progress):
                torch.save(net.state_dict(), path)
            with patch.object(models.utils, 'download_url_to_file',
                              side_effect=provide_checkpoint) as fetch:
                for selected in ['cpsam', 'cpsam_v2', str(local)]:
                    model = models.CellposeModel(pretrained_model=selected,
                        device=torch.device('cpu'), use_bfloat16=False)
                    assert isinstance(model.net, vit.CPSAM)
                    assert model.backbone == 'sam_vitl'
                assert fetch.call_count == 2
        assert finder.attempts == []
    """)
