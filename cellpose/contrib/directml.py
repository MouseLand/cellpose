# Created by https://github.com/Teranis while working on https://github.com/SchmollerLab/Cell_ACDC
# See below for working example

# Limitations:
# Officially support only up to PyTorch 2.4.1 (should be fine with cellpose)
# Not yet out for python 3.13
# Probably not the fastest option, but works suprisingly fast and was easy to implement

# Notes:
# No additional drivers needed, but requires Windows 10/11 and a DirectX 12 compatible GPU
# Install using "pip install torch-directml"

# Links:
# DirectML: https://microsoft.github.io/DirectML/  
# torch_directml: https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows


def setup_custom_device(model, device):
    """
    Forces the model to use a custom device (e.g., DirectML) for inference.
    This is a workaround, and could be handled better in the future. 
    (Ideally when all parameters are set initially)

    Args:
        model (cellpose.CellposeModel|cellpse.Cellpose): Cellpose model. Should work for v2, v3 and custom.
        torch.device (torch.device): Custom device.

    Returns:
        model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model with custom device set.
    """
    model.gpu = True
    model.device = device
    model.mkldnn = False
    if hasattr(model, 'net'):
        model.net.to(device)
        model.net.mkldnn = False
    if hasattr(model, 'cp'):
        model.cp.gpu = True
        model.cp.device = device
        model.cp.mkldnn = False
        if hasattr(model.cp, 'net'):
            model.cp.net.to(device)
            model.cp.net.mkldnn = False
    if hasattr(model, 'sz'):
        model.sz.device = device
    
    return model

def setup_directML(model):
    """
    Sets up the Cellpose model to use DirectML for inference.

    Args:
        model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model. Should work for v2, v3 and custom.
    
    Returns:
        model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model with DirectML set as the device.
    """
    print(
        'Using DirectML GPU for Cellpose model inference'
    )
    import torch_directml
    directml_device = torch_directml.device()
    model = setup_custom_device(model, directml_device)
    return model

def fix_sparse_directML(verbose=True):
    """DirectML does not support sparse tensors, so we need to fallback to CPU.
    This function replaces `torch.sparse_coo_tensor`, `torch._C._sparse_coo_tensor_unsafe`,
    `torch._C._sparse_coo_tensor_with_dims_and_tensors`, `torch.sparse.SparseTensor`
     with a wrapper that falls back to CPU.

    In the end, this could be handled better in the future. It would probably run faster if we
    just manually set the device to CPU, but my goal was to not modify the code too much,
    and this runs suprisingly fast.
    """
    import torch
    import functools
    import warnings

    def fallback_to_cpu_on_sparse_error(func, verbose=True):
        @functools.wraps(func) # wrapper shinanigans (thanks chatgpt)
        def wrapper(*args, **kwargs):
            device_arg = kwargs.get('device', None) # get desired device from kwargs

            # Ensure indices are int64 if args[0] looks like indices,
            # I got random errors from it not being int64
            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                if args[0].dtype != torch.int64:
                    args = (args[0].to(dtype=torch.int64),) + args[1:]

            try: # try to perform the operation and move to dml if possible
                result = func(*args, **kwargs) # run function with current args and kwargs
                if device_arg is not None and str(device_arg).lower() == "dml":
                    try: # try to move result to dml
                        result.to("dml")
                    except RuntimeError as e: # moving failed, falling back to cpu 
                        if verbose:
                            warnings.warn(f"Sparse op failed on DirectML, falling back to CPU: {e}")
                        kwargs['device'] = torch.device("cpu")
                        return func(*args, **kwargs) # try again, after setting device to cpu
                return result # just return result if all worked well

            except RuntimeError as e: # try and run on dlm, if it fails, fallback to cpu
                if "sparse" in str(e).lower() or "not implemented" in str(e).lower():
                    if verbose:
                        warnings.warn(f"Sparse op failed on DirectML, falling back to CPU: {e}")
                    kwargs['device'] = torch.device("cpu") # if rutime warning caused by sparse tensor, set device to cpu

                    # Re-apply indices dtype correction before retrying on CPU. Just in case (maybe first one not needed?)
                    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                        if args[0].dtype != torch.int64:
                            args = (args[0].to(dtype=torch.int64),) + args[1:]

                    return func(*args, **kwargs) # run function again with cpu device
                else:
                    raise e # catch and other runtime errors

        return wrapper

    # --- Patch Sparse Tensor Constructors ---

    # High-level API
    torch.sparse_coo_tensor = fallback_to_cpu_on_sparse_error(torch.sparse_coo_tensor, verbose=verbose)

    # Low-level API
    if hasattr(torch._C, "_sparse_coo_tensor_unsafe"):
        torch._C._sparse_coo_tensor_unsafe = fallback_to_cpu_on_sparse_error(torch._C._sparse_coo_tensor_unsafe, verbose=verbose)

    if hasattr(torch._C, "_sparse_coo_tensor_with_dims_and_tensors"):
        torch._C._sparse_coo_tensor_with_dims_and_tensors = fallback_to_cpu_on_sparse_error(
            torch._C._sparse_coo_tensor_with_dims_and_tensors, verbose=verbose
        )

    if hasattr(torch.sparse, 'SparseTensor'):
        torch.sparse.SparseTensor = fallback_to_cpu_on_sparse_error(torch.sparse.SparseTensor, verbose=verbose)
    
    # suppress warnings
    if not verbose:
        import warnings
        warnings.filterwarnings("once", message="Sparse op failed on DirectML*")



### (Not) working example
import cellpose
from cellpose import models
import time
# from cellpose.contrib.directml import setup_directML, fix_sparse_directML

# load data
path = r'path\to\your\images'  # path to your images
imgs = cellpose.io.imread(path)

# how to preprocess images?

# here goes the model  innit
model_path = r'path\to\your\model'  # path to your model
model = models.CellposeModel(
    pretrained_model=model_path,
)

uses_directml = False

# setup DirectML (comment out for performance comparison)
model = setup_directML(model)
fix_sparse_directML()
uses_directml = True

# run model
print("Running model...")
start = time.perf_counter()
out = model.eval(imgs,) # here goes the eval
end = time.perf_counter()

# show results

if uses_directml:
    print("DirectML inference completed.")
else:
    print("CPU inference completed.")

print(f"Time taken: {end - start:.2f} seconds")