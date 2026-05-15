"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.

Utilities for converting PyTorch Cellpose-SAM weights to MLX format.
"""

import logging
import numpy as np

mlx_logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def _torch_state_dict_to_numpy(path):
    """Load a PyTorch state dict and convert all tensors to numpy arrays.

    Args:
        path: Path to the PyTorch model file.

    Returns:
        dict: State dict with numpy arrays.
    """
    import torch
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Handle DataParallel/DistributedDataParallel wrapper
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Validate CP4 model
    if "W2" not in state_dict:
        raise ValueError(
            "This model does not appear to be a CP4 model. "
            "CP3 models are not compatible with CP4."
        )

    numpy_dict = {}
    for k, v in state_dict.items():
        numpy_dict[k] = v.float().numpy()
    return numpy_dict


def convert_pytorch_to_mlx_weights(state_dict, model):
    """Convert a PyTorch state dict (numpy arrays) to MLX model weight format.

    The key challenge is mapping PyTorch's nested module naming to MLX's
    tree-structured weight dictionary, and transposing Conv2d weights from
    PyTorch (O, I, H, W) to MLX (O, H, W, I) format.

    Args:
        state_dict: Dict of {pytorch_key: numpy_array}.
        model: The MLX TransformerMLX model instance.

    Returns:
        dict: Nested dictionary suitable for model.update().
    """
    weights = {}

    for pt_key, value in state_dict.items():
        mlx_path = _map_key(pt_key)
        if mlx_path is None:
            continue  # skip unmapped keys

        # Convert conv2d weights: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        arr = value
        if _is_conv_weight(pt_key, arr):
            arr = np.transpose(arr, (0, 2, 3, 1))

        _set_nested(weights, mlx_path, mx.array(arr))

    # MLX model.update() expects Python lists for list-based submodules
    # (e.g., self.blocks = [Block(...), ...]).  _set_nested builds plain
    # dicts with string keys like {"0": {...}, "1": {...}}.  Convert any
    # such dict whose keys are consecutive integers starting at 0 into a
    # list so that update() can match them by index.
    weights = _dicts_to_lists(weights)

    return weights


def _is_conv_weight(key, arr):
    """Check if a weight tensor is a Conv2d weight (4D, not W2)."""
    if arr.ndim != 4:
        return False
    if key == "W2":
        return False
    # Conv weights in the model
    conv_patterns = [
        "encoder.patch_embed.proj.weight",
        "encoder.neck.0.weight",  # conv1
        "encoder.neck.2.weight",  # conv2
        "out.weight",             # readout conv
    ]
    return key in conv_patterns


def _map_key(pt_key):
    """Map a PyTorch state dict key to an MLX nested key path.

    Returns:
        list of str path components, or None to skip.
    """
    # Skip W2 - it's a fixed identity matrix, reconstructed in __init__
    if pt_key == "W2":
        return None

    # Diameter parameters
    if pt_key == "diam_labels":
        return ["diam_labels"]
    if pt_key == "diam_mean":
        return ["diam_mean"]

    # Patch embedding
    if pt_key == "encoder.patch_embed.proj.weight":
        return ["patch_embed", "proj", "weight"]
    if pt_key == "encoder.patch_embed.proj.bias":
        return ["patch_embed", "proj", "bias"]

    # Positional embedding
    if pt_key == "encoder.pos_embed":
        return ["pos_embed"]

    # Transformer blocks: encoder.blocks.{i}.{submodule}
    if pt_key.startswith("encoder.blocks."):
        rest = pt_key[len("encoder.blocks."):]
        parts = rest.split(".", 1)
        block_idx = parts[0]
        sub_key = parts[1]
        return ["blocks", block_idx] + _map_block_key(sub_key)

    # Neck
    # encoder.neck.0 = conv1, encoder.neck.1 = ln1, encoder.neck.2 = conv2, encoder.neck.3 = ln2
    if pt_key.startswith("encoder.neck."):
        rest = pt_key[len("encoder.neck."):]
        parts = rest.split(".", 1)
        neck_idx = int(parts[0])
        param = parts[1]
        neck_map = {0: "conv1", 1: "ln1", 2: "conv2", 3: "ln2"}
        return ["neck", neck_map[neck_idx], param]

    # Readout convolution
    if pt_key == "out.weight":
        return ["out_conv", "weight"]
    if pt_key == "out.bias":
        return ["out_conv", "bias"]

    mlx_logger.warning(f"Unmapped PyTorch key: {pt_key}")
    return None


def _map_block_key(sub_key):
    """Map block sub-keys from PyTorch to MLX format.

    PyTorch: norm1.weight, attn.qkv.weight, attn.proj.weight, norm2.weight, mlp.lin1.weight, etc.
    MLX: same structure, just different nesting.
    """
    # norm1, norm2
    if sub_key.startswith("norm1."):
        param = sub_key[len("norm1."):]
        return ["norm1", param]
    if sub_key.startswith("norm2."):
        param = sub_key[len("norm2."):]
        return ["norm2", param]

    # attention: attn.qkv.weight, attn.qkv.bias, attn.proj.weight, attn.proj.bias
    if sub_key.startswith("attn."):
        attn_rest = sub_key[len("attn."):]
        parts = attn_rest.split(".")
        return ["attn"] + parts

    # MLP: mlp.lin1.weight, mlp.lin1.bias, mlp.lin2.weight, mlp.lin2.bias
    if sub_key.startswith("mlp."):
        mlp_rest = sub_key[len("mlp."):]
        parts = mlp_rest.split(".")
        return ["mlp"] + parts

    mlx_logger.warning(f"Unmapped block sub-key: {sub_key}")
    return [sub_key]


def _set_nested(d, path, value):
    """Set a value in a nested dictionary using a list of keys."""
    for key in path[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[path[-1]] = value


def _dicts_to_lists(obj):
    """Recursively convert dicts with consecutive-integer string keys to lists.

    MLX's ``Module.update()`` expects Python lists for list-based submodules
    (e.g. ``self.blocks``).  The weight-building step produces plain dicts
    with string keys ``{"0": ..., "1": ...}`` which must be converted.
    """
    if not isinstance(obj, dict):
        return obj

    # Recurse first so children are converted before we inspect keys
    converted = {k: _dicts_to_lists(v) for k, v in obj.items()}

    # Check if all keys are consecutive integers starting at 0
    if all(k.isdigit() for k in converted):
        indices = sorted(int(k) for k in converted)
        if indices == list(range(len(indices))):
            return [converted[str(i)] for i in indices]

    return converted


def save_mlx_weights(state_dict_path, output_path):
    """Convert a PyTorch checkpoint to MLX safetensors format.

    Args:
        state_dict_path: Path to PyTorch model file.
        output_path: Path to save MLX weights (.safetensors or .npz).
    """
    from .mlx_net import TransformerMLX
    numpy_dict = _torch_state_dict_to_numpy(state_dict_path)
    model = TransformerMLX()
    mlx_weights = convert_pytorch_to_mlx_weights(numpy_dict, model)

    # Flatten nested dict for saving
    flat = {}
    _flatten_dict(mlx_weights, [], flat)

    if output_path.endswith(".npz"):
        np_dict = {k: np.array(v) for k, v in flat.items()}
        np.savez(output_path, **np_dict)
    else:
        mx.savez(output_path, **flat)

    mlx_logger.info(f"MLX weights saved to {output_path}")


def _flatten_dict(d, prefix, out):
    """Flatten a nested dict to dot-separated keys."""
    for k, v in d.items():
        key = ".".join(prefix + [k])
        if isinstance(v, dict):
            _flatten_dict(v, prefix + [k], out)
        else:
            out[key] = v
