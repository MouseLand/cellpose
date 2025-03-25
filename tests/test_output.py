from cellpose import io, models, metrics, plot, utils
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import numpy as np
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
            ext = ".png"
        else:
            cached_file = str(data_dir_3D.joinpath(image_name))
            ext = ".tif"
        name, ext = os.path.splitext(cached_file)
        output = name + "_cp_masks" + ext
        if os.path.exists(output):
            os.remove(output)

def test_class_2D(data_dir, image_names):
    clear_output(data_dir, image_names)
    image_name = "rgb_2D.png"
    img = io.imread(str(data_dir.joinpath("2D").joinpath(image_name)))
    model_types = ["nuclei"]
    chan = [1]
    chan2 = [0]
    for m, model_type in enumerate(model_types):
        model = models.Cellpose(model_type=model_type)
        masks, flows, _, _ = model.eval(img, diameter=0, cellprob_threshold=0,
                                        channels=[chan[m], chan2[m]], resample=False)
        io.imsave(str(data_dir.joinpath("2D").joinpath("rgb_2D_cp_masks.png")), masks)
        compare_masks(data_dir, [image_name], "2D", model_type)
        clear_output(data_dir, image_names)
        if MATPLOTLIB:
            fig = plt.figure(figsize=(8, 3))
            plot.show_segmentation(fig, img, masks, flows[0],
                                   channels=[chan[m], chan2[m]])


def test_cyto2_to_seg(data_dir, image_names):
    clear_output(data_dir, image_names)
    image_names = ["rgb_2D.png", "rgb_2D_tif.tif"]
    file_names = [
        str(data_dir.joinpath("2D").joinpath(image_name)) for image_name in image_names
    ]
    imgs = [io.imread(file_name) for file_name in file_names]
    model_type = "cyto2"
    model = models.Cellpose(model_type=model_type)
    channels = [2, 1]
    masks, flows, styles, diams = model.eval(imgs, diameter=30, channels=channels)
    io.masks_flows_to_seg(imgs, masks, flows, file_names, diams=diams)


def test_class_3D(data_dir, image_names):
    clear_output(data_dir, image_names)
    img = io.imread(str(data_dir.joinpath("3D").joinpath("rgb_3D.tif")))
    model_types = ["nuclei"]
    chan = [1]
    chan2 = [0]
    for m, model_type in enumerate(model_types):
        model = models.Cellpose(model_type="nuclei")
        masks = model.eval(img, do_3D=True, diameter=25, 
                           channels=[chan[m], chan2[m]], resample=True)[0]
        io.imsave(str(data_dir.joinpath("3D").joinpath("rgb_3D_cp_masks.tif")), masks)
        compare_masks(data_dir, ["rgb_3D.tif"], "3D", model_type)
        clear_output(data_dir, image_names)


def test_cli_2D(data_dir, image_names):
    clear_output(data_dir, image_names)
    model_types = ["cyto"]
    chan = [2]
    chan2 = [1]
    for m, model_type in enumerate(model_types):
        cmd = "python -m cellpose --dir %s --pretrained_model %s --no_resample --chan %d --chan2 %d --diameter 0 --save_png --verbose" % (
            str(data_dir.joinpath("2D")), model_type, chan[m], chan2[m])
        try:
            cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
            print(cmd_stdout)
        except Exception as e:
            print(e)
            raise ValueError(e)
        compare_masks(data_dir, image_names, "2D", model_type)
    clear_output(data_dir, image_names)


def test_cli_3D(data_dir, image_names):
    clear_output(data_dir, image_names)
    model_types = ["cyto"]
    chan = [2]
    chan2 = [1]
    for m, model_type in enumerate(model_types):
        cmd = "python -m cellpose --dir %s --do_3D --pretrained_model %s --no_resample --cellprob_threshold 0 --chan %d --chan2 %d --diameter 25 --save_tif" % (
            str(data_dir.joinpath("3D")), model_type, chan[m], chan2[m])
        try:
            cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
        except Exception as e:
            print(e)
            raise ValueError(e)
        compare_masks(data_dir, image_names, "3D", model_type)
        clear_output(data_dir, image_names)

def test_outlines_list(data_dir, image_names):
    """ test both single and multithreaded by comparing them"""
    clear_output(data_dir, image_names)
    model_type = "cyto"
    channels = [2, 1]
    image_name = "rgb_2D.png"

    file_name = str(data_dir.joinpath("2D").joinpath(image_name))
    img = io.imread(file_name)

    model = models.Cellpose(model_type=model_type)
    masks, _, _, _ = model.eval(img, diameter=30, channels=channels)
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


def compare_masks(data_dir, image_names, runtype, model_type):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    data_dir_2D = data_dir.joinpath("2D")
    data_dir_3D = data_dir.joinpath("3D")
    for image_name in image_names:
        check = False
        if "2D" in runtype and "2D" in image_name:
            image_file = str(data_dir_2D.joinpath(image_name))
            name = os.path.splitext(image_file)[0]
            output_test = name + "_cp_masks.png"
            output_true = name + "_%s_masks.png" % model_type
            check = True
        elif "3D" in runtype and "3D" in image_name:
            image_file = str(data_dir_3D.joinpath(image_name))
            name = os.path.splitext(image_file)[0]
            output_test = name + "_cp_masks.tif"
            output_true = name + "_%s_masks.tif" % model_type
            check = True

        if check:
            if os.path.exists(output_test):
                print("checking output %s" % output_test)
                masks_test = io.imread(output_test)
                masks_true = io.imread(output_true)
                print("masks", np.unique(masks_test), np.unique(masks_true),
                      output_test, output_true)
                thresholds = [0.5, 0.75, 0.9]
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
