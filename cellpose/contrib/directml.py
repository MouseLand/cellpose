# Created by https://github.com/Teranis while working on https://github.com/SchmollerLab/Cell_ACDC
# See below for working example

# Limitations:
# Officially support only up to PyTorch 2.4.1 (should be fine with cellpose)
# Not yet out for python 3.13
# Probably not the fastest option, but works surprisingly fast and was easy to implement

# Notes:
# No additional drivers needed, but requires Windows 10/11 and a DirectX 12 compatible GPU
# Install using "pip install torch-directml"

# Links:
# DirectML: https://microsoft.github.io/DirectML/  
# torch_directml: https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows

# Examples:
# Entire working example with benchmark and save comparison is at the end of this file

# Example usage:
# from cellpose import models as models
# model = models.CellposeModel(gpu=True)
# out = model.eval(img)




### This function has been made obsolete by updates to cellpose.models
# def setup_custom_device(model, device):
#     """
#     Forces the model to use a custom device (e.g., DirectML) for inference.
#     This is a workaround, and could be handled better in the future. 
#     (Ideally when all parameters are set initially)

#     Args:
#         model (cellpose.CellposeModel|cellpse.Cellpose): Cellpose model. Should work for v2, v3 and custom.
#         torch.device (torch.device): Custom device.

#     Returns:
#         model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model with custom device set.
#     """
#     model.gpu = True
#     model.device = device
#     model.mkldnn = False
#     if hasattr(model, 'net'):
#         model.net.to(device)
#         model.net.mkldnn = False
#     if hasattr(model, 'cp'):
#         model.cp.gpu = True
#         model.cp.device = device
#         model.cp.mkldnn = False
#         if hasattr(model.cp, 'net'):
#             model.cp.net.to(device)
#             model.cp.net.mkldnn = False
#     if hasattr(model, 'sz'):
#         model.sz.device = device
    
#     return model

# def setup_directML(model):
#     """
#     Sets up the Cellpose model to use DirectML for inference.

#     Args:
#         model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model. Should work for v2, v3 and custom.
    
#     Returns:
#         model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model with DirectML set as the device.
#     """
#     print(
#         'Using DirectML GPU for Cellpose model inference'
#     )
#     import torch_directml
#     directml_device = torch_directml.device()
#     model = setup_custom_device(model, directml_device)
#     return model

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
            # If errors start to occur that int64 conversion is needed, uncomment this
            # (and also consider the block below).
            # But be aware! Its probably better to just set the device to cpu in that 
            # particular case...
            # for both performance and compatibility
            # if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            #     if args[0].dtype != torch.int64:
            #         args = (args[0].to(dtype=torch.int64),) + args[1:]

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

                    # See above comments
                    # if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                    #     if args[0].dtype != torch.int64:
                    #         args = (args[0].to(dtype=torch.int64),) + args[1:]
                    try:
                        res = func(*args, **kwargs)
                    except RuntimeError as e: # try again with cpu device
                        if "int64" in str(e).lower():
                            if verbose:
                                warnings.warn(f"need to convert to int64: {e}")
                            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                                if args[0].dtype != torch.int64:
                                    args = (args[0].to(dtype=torch.int64),) + args[1:]
                            return func(*args, **kwargs)
                    return res # run function again with cpu device
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

if __name__ == "__main__":
    import time
    import numpy as np
    import tifffile
    import os

    ### Working example with benchmark and save comparison
    def _load_data(path, prepare):
        """
        Load and prepare data for Cellpose model.
        Args:
            path (str): Path to the image data.
            prepare (bool): Whether to prepare the data for Cellpose model.
        Returns:
            imgs_list (list): List of images prepared for Cellpose model.
        """

        # load data
        imgs = tifffile.imread(path)  # read images using tifffile
        print(imgs.shape)
        if prepare:
            imgs_list = []
            for img in imgs:  # convert to list of images
                img_min = img.min() 
                img_max = img.max()
                img = img.astype(np.float32)
                img -= img_min
                if img_max > img_min + 1e-3:
                    img /= (img_max - img_min)
                img *= 255
                
                img = img.astype(np.float32)
                imgs_list.append(img)  # add image to list
        
            return imgs_list
        else:
            return imgs

    def _compare_data(savepaths):
        """
        Compare data from different save paths to check for consistency.
        Args:
            savepaths (list): List of paths to the saved data.
        """
        outs = dict()
        for savepath in savepaths:
            if not os.path.exists(savepath):
                continue
            out = np.load(savepath)
            out = out[out.files[0]]
            outs[savepath] = out

        total_size = out.shape[1] * out.shape[2]
        last_out = None
        for savepath, out in outs.items():
            file_name = os.path.basename(savepath)
            mismatch = False
            if last_out is None:
                last_out = out
                last_file_name = file_name
                continue
            if out.shape != last_out.shape:
                print(f"Shape mismatch for {file_name} vs {last_file_name}: {out.shape} vs {last_out.shape}")
                continue

            for frame in range(out.shape[0]):
                seg_difference = np.nonzero(out[frame] - last_out[frame])
                perc_diff = len(seg_difference[0]) / total_size
                if perc_diff > 0.01:
                    print(f"Frame {frame} mismatch for {file_name} vs {last_file_name} with {perc_diff:.2%} difference")
                    mismatch = True
            
            if not mismatch:
                print(f"All frames match for {file_name} vs {last_file_name}")


    # you need two environment for benchmarking: One with DirectML and one with CUDA.
    path = r'path\to\your\data.tif'  # path to your data
    # pretrained_model = r'path\to\your\model'  # path to your pretrained model
    pretrained_model = "cpsam" # "cyto3" # for pretrained models
    gpu = True  # set to True if you want to use GPU
    # if False, CPU will be used
    just_compare_data = False # set to True if you want to compare data and exit

    # load and prepare images
    imgs = _load_data(path, prepare=True) 
    imgs = imgs[:10] # cut data so we can test it faster

    # save paths for different methods (Don't change order!)
    savepaths = [
        path.replace('.tif', '_segm_directml.npz'),
        path.replace('.tif', '_segm_GPU.npz'),
        path.replace('.tif', '_segm_CPU.npz')
    ]

    # for data comparison
    if just_compare_data:
        _compare_data(savepaths)
        exit()

    # init model
    from cellpose import models, io
    io.logger_setup()
    model = models.CellposeModel(
        pretrained_model=pretrained_model, gpu=gpu, 
    )

    # run model, benchmark
    print("Running model...")
    start = time.perf_counter()
    pref_count_last = time.perf_counter()
    times = []
    out_list = []
    for img in imgs: # process each image
        out = model.eval(img)[0] # here goes the eval
        out_list.append(out)
        time_taken = time.perf_counter() - pref_count_last
        times.append(time_taken)
        print(f'processed image in {time_taken:.2f} seconds')
        pref_count_last = time.perf_counter()
    end = time.perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Average time per image: {np.mean(times):.2f} seconds")

    uses_directml = model.device.type == 'privateuseone'
    # save data
    if uses_directml: 
        print("DirectML inference completed.")
        savepath = savepaths[0]
    elif gpu:
        print("GPU inference completed.")
        savepath = savepaths[1]
    else:
        print("CPU inference completed.")
        savepath = savepaths[2]

    np.savez_compressed(savepath, out_list=out_list)