from cellpose import io, metrics, utils, models
import pytest
from subprocess import check_output, STDOUT
import os
import numpy as np

import torch

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

r_tol, a_tol = 1e-2, 1e-2


def clear_output(data_dir, image_names):
    data_dir_2D = data_dir.joinpath("2D")
    data_dir_3D = data_dir.joinpath("2D")
    for image_name in image_names:
        if "2D" in image_name:
            cached_file = str(data_dir_2D.joinpath(image_name))
        else:
            cached_file = str(data_dir_3D.joinpath(image_name))
        name, _ = os.path.splitext(cached_file)
        output_png = name + "_cp_masks.png"
        if os.path.exists(output_png):
            os.remove(output_png)
        output_tif = name + "_cp_masks.tif"
        if os.path.exists(output_tif):
            os.remove(output_tif)
        flowp_output = name + "_flowps_test.tif"
        if os.path.exists(flowp_output):
            os.remove(flowp_output)
        npy_output = name + "_seg.npy"
        if os.path.exists(npy_output):
            os.remove(npy_output)


@pytest.mark.parametrize('compute_masks, resample, diameter', 
                         [
                             (True, True, 40), 
                             (True, True, None), 
                             (False, True, None),
                             (False, False, None)
                         ]
)
def test_class_2D_one_img(data_dir, image_names, cellposemodel_fixture_24layer, compute_masks, resample, diameter):
    clear_output(data_dir, image_names)
        
    img_file = data_dir / '2D' / image_names[0]

    img = io.imread_2D(img_file)
    # flowps = io.imread(img_file.parent / (img_file.stem + "_cp4_gt_flowps.tif"))

    masks_pred, _, _ = cellposemodel_fixture_24layer.eval(img, normalize=True, compute_masks=compute_masks, resample=resample, diameter=diameter)

    if not compute_masks or diameter:
        # not compute_masks won't return masks so can't check
        # different diameter will give different masks, so can't check
        return 
    
    io.imsave(data_dir / '2D' / (img_file.stem + "_cp_masks.png"), masks_pred)
    # flowsp_pred = np.concatenate([flows_pred[1], flows_pred[2][None, ...]], axis=0)
    # mse = np.sqrt((flowsp_pred - flowps) ** 2).sum()
    # assert mse.sum() < 1e-8, f"MSE of flows is too high: {mse.sum()} on image {image_name}"
    # print("MSE of flows is %f" % mse.mean())

    compare_masks_cp4(data_dir, image_names[0], "2D")
    # clear_output(data_dir, image_names)


@pytest.mark.slow
def test_class_2D_all_imgs(data_dir, image_names, cellposemodel_fixture_24layer):
    clear_output(data_dir, image_names)
    for image_name in image_names:
        img_file = data_dir / '2D' / image_name

        img = io.imread_2D(img_file)
        # flowps = io.imread(img_file.parent / (img_file.stem + "_cp4_gt_flowps.tif"))

        masks_pred, _, _ = cellposemodel_fixture_24layer.eval(img, normalize=True)
        io.imsave(data_dir / '2D' / (img_file.stem + "_cp_masks.png"), masks_pred)
        # flowsp_pred = np.concatenate([flows_pred[1], flows_pred[2][None, ...]], axis=0)
        # mse = np.sqrt((flowsp_pred - flowps) ** 2).sum()
        # assert mse.sum() < 1e-8, f"MSE of flows is too high: {mse.sum()} on image {image_name}"
        # print("MSE of flows is %f" % mse.mean())

    compare_masks_cp4(data_dir, image_names, "2D")
    clear_output(data_dir, image_names)


@pytest.mark.slow
def test_flows_to_seg(data_dir, image_names, cellposemodel_fixture_24layer):
    clear_output(data_dir, image_names)
    file_names = [data_dir / "2D" / n for n in image_names]
    imgs = [io.imread_2D(file_name) for file_name in file_names]

    # masks, flows, styles = model.eval(imgs, diameter=30)  # Errors during SAM stuff
    masks, flows, _ = cellposemodel_fixture_24layer.eval(imgs, bsize=256, batch_size=64, normalize=True)

    for file_name, mask in zip(file_names, masks):
        io.imsave(data_dir/'2D'/(file_name.stem + '_cp_masks.png'), mask)

    io.masks_flows_to_seg(imgs, masks, flows, file_names)
    compare_masks_cp4(data_dir, image_names, "2D")
    clear_output(data_dir, image_names)


def test_class_3D_one_img_shape(data_dir, image_names_3d, cellposemodel_fixture_2layer):
    clear_output(data_dir, image_names_3d)

    img_file = data_dir / '3D' / image_names_3d[0]
    image_name = img_file.name
    img = io.imread_3D(img_file)
    masks_pred, flows_pred, _ = cellposemodel_fixture_2layer.eval(img, do_3D=True, channel_axis=-1, z_axis=0)

    assert img.shape[:-1] == masks_pred.shape, f'mask incorrect shape for {image_name}, {masks_pred.shape=}'
    assert img.shape[:-1] == flows_pred[1].shape[1:], f'flows incorrect shape for {image_name}, {flows_pred.shape=}'

    # just compare shapes for now
    # compare_masks_cp4(data_dir, image_names_3d, "3D")
    clear_output(data_dir, image_names_3d)


def test_cli_2D(data_dir, image_names):
    clear_output(data_dir, image_names)
    use_gpu = torch.cuda.is_available()
    gpu_string = "--use_gpu" if use_gpu else ""
    image_path_string = str(data_dir/"2D"/image_names[0])
    cmd = f"python -m cellpose --image_path {image_path_string} --save_png --verbose {gpu_string}"
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
        print(cmd_stdout)
    except Exception as e:
        print(e)
        raise ValueError(e)
    compare_masks_cp4(data_dir, image_names[0], "2D")
    clear_output(data_dir, image_names)


@pytest.mark.parametrize('diam, aniso', [(None, 2.5), (25, 2.5), (25, None)])
@pytest.mark.slow
def test_cli_3D_diam_anisotropy_shape(data_dir, image_names_3d, diam, aniso):
    clear_output(data_dir, image_names_3d)
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available() 
    gpu_string = "--use_gpu" if use_gpu else ""
    anisotropy_text = f" {'--anisotropy ' + str(aniso) if aniso else ''}"
    diam_text = f" {'--diameter ' + str(diam) if diam else ''}"
    cmd = f"python -m cellpose --image_path {str(data_dir / '3D' / image_names_3d[0])} --do_3D --save_tif {gpu_string} --verbose" + anisotropy_text + diam_text
    print(cmd)
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
        print(cmd_stdout)
    except Exception as e:
        print(e)
        raise ValueError(e)
    compare_mask_shapes(data_dir, image_names_3d[0], "3D")
    clear_output(data_dir, image_names_3d)


@pytest.mark.slow
def test_cli_3D_one_img(data_dir, image_names_3d):
    clear_output(data_dir, image_names_3d)
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available() 
    gpu_string = "--use_gpu" if use_gpu else ""
    cmd = f"python -m cellpose --image_path {str(data_dir / '3D' / image_names_3d[0])} --do_3D --save_tif {gpu_string} --verbose"
    print(cmd)
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
        print(cmd_stdout)
    except Exception as e:
        print(e)
        raise ValueError(e)
    compare_masks_cp4(data_dir, image_names_3d[0], "3D")
    clear_output(data_dir, image_names_3d)


@pytest.mark.slow
def test_outlines_list(data_dir, image_names, cellposemodel_fixture_24layer):
    """ test both single and multithreaded by comparing them"""
    clear_output(data_dir, image_names)
    image_name = "rgb_2D.png"

    file_name = str(data_dir.joinpath("2D").joinpath(image_name))
    img = io.imread(file_name)

    masks, _, _ = cellposemodel_fixture_24layer.eval(img, diameter=30)
    outlines_single = utils.outlines_list(masks, multiprocessing=False)
    outlines_multi = utils.outlines_list(masks, multiprocessing=True)

    assert len(outlines_single) == len(outlines_multi)

    # Check that the outlines are the same, but not necessarily in the same order
    outlines_matched = [False] * len(outlines_single)
    for i, outline_single in enumerate(outlines_single):
        for j, outline_multi in enumerate(outlines_multi):
            if not outlines_matched[j] and np.array_equal(outline_single,
                                                          outline_multi):
                outlines_matched[j] = True
                break
        else:
            assert False, "Outline not found in outlines_multi: {}".format(
                outline_single)

    assert all(outlines_matched), "Not all outlines in outlines_multi were matched"


def compare_masks_cp4(data_dir, image_names, runtype):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    data_dir_2D = data_dir.joinpath("2D")
    data_dir_3D = data_dir.joinpath("3D")
    if not isinstance(image_names, list):
        image_names = [image_names]
    for image_name in image_names:
        check = False
        if "2D" in runtype and "2D" in image_name:
            image_file = str(data_dir_2D.joinpath(image_name))
            name = os.path.splitext(image_file)[0]
            output_test = name + "_cp_masks.png"
            output_true = name + "_cp4_gt_masks.png"
            check = True
        elif "3D" in runtype and "3D" in image_name:
            image_file = str(data_dir_3D.joinpath(image_name))
            name = os.path.splitext(image_file)[0]
            output_test = name + "_cp_masks.tif"
            output_true = name + "_cp4_gt_masks.tif"
            check = True

        if check:
            if os.path.exists(output_test):
                print("checking output %s" % output_test)
                masks_test = io.imread(output_test)
                masks_true = io.imread(output_true)
                print("masks", np.unique(masks_test), np.unique(masks_true),
                      output_test, output_true)
                thresholds = [0.5, 0.75]
                ap = metrics.average_precision(masks_true, masks_test, 
                                               threshold=thresholds)[0]
                print("average precision of ", ap)
                ap_precision = np.allclose(ap, np.ones(len(thresholds)), 
                                           rtol=r_tol, atol=a_tol)

                matching_pix = np.logical_and(masks_test > 0, masks_true > 0).mean()
                all_pix = (masks_test > 0).mean()
                pix_precision = np.allclose(all_pix, matching_pix, rtol=r_tol,
                                            atol=a_tol)
                assert all([ap_precision, pix_precision])
            else:
                print("ERROR: no output file of name %s found" % output_test)
                assert False


def compare_mask_shapes(data_dir, image_names, runtype):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    data_dir_2D = data_dir.joinpath("2D")
    data_dir_3D = data_dir.joinpath("3D")
    if not isinstance(image_names, list):
        image_names = [image_names]
    for image_name in image_names:
        check = False
        if "2D" in runtype and "2D" in image_name:
            image_file = str(data_dir_2D.joinpath(image_name))
            name = os.path.splitext(image_file)[0]
            output_test = name + "_cp_masks.png"
            output_true = name + "_cp4_gt_masks.png"
            check = True
        elif "3D" in runtype and "3D" in image_name:
            image_file = str(data_dir_3D.joinpath(image_name))
            name = os.path.splitext(image_file)[0]
            output_test = name + "_cp_masks.tif"
            output_true = name + "_cp4_gt_masks.tif"
            check = True

        if check:
            if os.path.exists(output_test):
                print("checking output %s" % output_test)
                masks_test = io.imread(output_test)
                masks_true = io.imread(output_true)

                assert all([a == b for a, b in zip(masks_test.shape, masks_true.shape)]), f'mask shape mismatch: {masks_test.shape} =/= {masks_true.shape}'
            else:
                print("ERROR: no output file of name %s found" % output_test)
                assert False
