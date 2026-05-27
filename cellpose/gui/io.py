"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os, gc
import numpy as np
import cv2
import fastremap

from ..io import imread, imread_2D, imread_3D, imsave, outlines_to_text, add_model, remove_model, save_rois
from ..models import normalize_default, MODEL_DIR, MODEL_LIST_PATH, get_user_models
from ..utils import masks_to_outlines, outlines_list

try:
    import qtpy
    from qtpy.QtWidgets import QFileDialog
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


def _init_model_list(parent):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = MODEL_LIST_PATH
    parent.model_strings = get_user_models()


def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Add model to GUI")
        filename = name[0]
    add_model(filename)
    fname = os.path.split(filename)[-1]
    parent.ModelChooseC.addItems([fname])
    parent.model_strings.append(fname)

    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind + 1)

    parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
    if load_model:
        parent.model_choose(custom=True)


def _remove_model(parent, ind=None):
    if ind is None:
        ind = parent.ModelChooseC.currentIndex()
    if ind > 0:
        modelstr = parent.ModelChooseC.currentText()
        parent.ModelChooseC.removeItem(ind)
        # remove model from txt path
        remove_model(modelstr)
        parent.model_strings.remove(modelstr)
        if len(parent.model_strings) > 0:
            parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
        else:
            parent.ModelChooseC.setCurrentIndex(0)
    else:
        print("ERROR: no model selected to delete")


def _get_train_set(image_names):
    """ get training data and labels for images in current folder image_names"""
    train_data, train_labels, train_files = [], [], []
    restore = None
    normalize_params = normalize_default
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        label_name = None
        if os.path.exists(image_name + "_seg.npy"):
            dat = np.load(image_name + "_seg.npy", allow_pickle=True).item()
            masks = dat["masks"].squeeze()
            if masks.ndim == 2:
                fastremap.renumber(masks, in_place=True)
                label_name = image_name + "_seg.npy"
            else:
                print(f"GUI_INFO: _seg.npy found for {image_name} but masks.ndim!=2")
            if "img_restore" in dat:
                data = dat["img_restore"].squeeze()
                restore = dat["restore"]
            else:
                data = imread(image_name_full)
            normalize_params = dat[
                "normalize_params"] if "normalize_params" in dat else normalize_default
        if label_name is not None:
            train_files.append(image_name_full)
            train_data.append(data)
            train_labels.append(masks)
    if restore:
        print(f"GUI_INFO: using {restore} images (dat['img_restore'])")
    return train_data, train_labels, train_files, restore, normalize_params


def _load_image(parent, filename=None, load_seg=True, load_3D=False):
    """ load image with filename; if None, open QFileDialog
    if image is grey change view to default to grey scale 
    """

    if parent.load_3D:
        load_3D = True

    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load image")
        filename = name[0]
        if filename == "":
            return
    manual_file = os.path.splitext(filename)[0] + "_seg.npy"
    load_mask = False
    if load_seg:
        if os.path.isfile(manual_file) and not parent.autoloadMasks.isChecked():
            if filename is not None:
                image = (imread_2D(filename) if not load_3D else 
                         imread_3D(filename))
            else:
                image = None
            _load_seg(parent, manual_file, image=image, image_file=filename,
                        load_3D=load_3D)
            if len(np.unique(image[..., 1:])) == 1:
                parent.color = 'gray' # updates the plot automatically
            else:
                parent.update_plot()
            return
        elif parent.autoloadMasks.isChecked():
            mask_file = os.path.splitext(filename)[0] + "_masks" + os.path.splitext(
                filename)[-1]
            mask_file = os.path.splitext(filename)[
                0] + "_masks.tif" if not os.path.isfile(mask_file) else mask_file
            load_mask = True if os.path.isfile(mask_file) else False
    try:
        print(f"GUI_INFO: loading image: {filename}")
        if not load_3D:
            image = imread_2D(filename)
        else:
            image = imread_3D(filename)
        parent.loaded = True
    except Exception as e:
        print("ERROR: images not compatible")
        print(f"ERROR: {e}")
        return

    if parent.loaded:
        parent.reset()
        parent.filename = filename
        filename = os.path.split(parent.filename)[-1]
        _initialize_images(parent, image, load_3D=load_3D)
        parent.loaded = True
        parent.enable_buttons()
        if load_mask:
            _load_masks(parent, filename=mask_file)

    # check if gray and adjust viewer:
    if len(np.unique(image[..., 1:])) == 1:
        parent.color = 'gray' # triggers update_plot
    else:
        parent.update_plot()

def _initialize_images(parent, image, load_3D=False):
    """ format image for GUI

    assumes image is Z x W x H x C

    """
    load_3D = parent.load_3D if load_3D is False else load_3D

    parent.stack = image
    parent.logger.info(f" : image shape: {image.shape}")
    if load_3D:
        parent.NZ = len(parent.stack)
        parent.scroll.setMaximum(parent.NZ - 1)
    else:
        parent.NZ = 1
        parent.stack = parent.stack[np.newaxis, ...]

    img_min = image.min()
    img_max = image.max()
    parent.stack = parent.stack.astype("float32")
    parent.stack -= img_min
    if img_max > img_min + 1e-3:
        parent.stack /= (img_max - img_min)
    parent.stack *= 255

    if parent.stack.shape[-1] == 2:
        parent.stack = np.concatenate((parent.stack, np.zeros((*parent.stack.shape[:-1], 1), dtype="float32")), axis=-1)
    print(parent.stack.shape)

    if load_3D:
        parent.logger.info(": converted to float and normalized values to 0.0->255.0")

    del image
    gc.collect()

    parent.imask = 0
    parent.Ly, parent.Lx = parent.stack.shape[-3:-1]
    parent.Ly0, parent.Lx0 = parent.stack.shape[-3:-1]
    parent.nchan = parent.stack.shape[-1]
    parent.layerz = 255 * np.ones((parent.Ly, parent.Lx, 4), "uint8")
    parent.Lyr, parent.Lxr = parent.stack.shape[-3:-1]
    parent.clear_all()

    parent.compute_saturation()    
    parent.compute_scale()
    parent.track_changes = []

    if load_3D:
        parent.currentZ = int(np.floor(parent.NZ / 2))
        parent.scroll.setValue(parent.currentZ)
        parent.zpos.setText(str(parent.currentZ))
    else:
        parent.currentZ = 0


def _load_seg(parent, filename=None, image=None, image_file=None, load_3D=False):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load labelled data", filter="*.npy")
        filename = name[0]
    try:
        dat = np.load(filename, allow_pickle=True).item()
        # check if there are keys in filename
        dat["outlines"]
        parent.loaded = True
    except:
        parent.loaded = False
        print("ERROR: not NPY")
        return

    parent.reset()
    if image is None:
        found_image = False
        if "filename" in dat:
            parent.filename = dat["filename"]
            if os.path.isfile(parent.filename):
                parent.filename = dat["filename"]
                found_image = True
            else:
                imgname = os.path.split(parent.filename)[1]
                root = os.path.split(filename)[0]
                parent.filename = root + "/" + imgname
                if os.path.isfile(parent.filename):
                    found_image = True
        if found_image:
            try:
                print(parent.filename)
                image = (imread_2D(parent.filename) if not load_3D else 
                         imread_3D(parent.filename))
            except:
                parent.loaded = False
                found_image = False
                print("ERROR: cannot find image file, loading from npy")
        if not found_image:
            parent.filename = filename[:-8]
            print(parent.filename)
            if "img" in dat:
                image = dat["img"]
            else:
                print("ERROR: no image file found and no image in npy")
                return
    else:
        parent.filename = image_file

    parent.restore = None
    parent.ratio = 1.

    if "normalize_params" in dat:
        parent.set_normalize_params(dat["normalize_params"])

    _initialize_images(parent, image, load_3D=load_3D)
    print(parent.stack.shape)

    if "outlines" in dat:
        if isinstance(dat["outlines"], list):
            # old way of saving files
            dat["outlines"] = dat["outlines"][::-1]
            for k, outline in enumerate(dat["outlines"]):
                if "colors" in dat:
                    color = dat["colors"][k]
                else:
                    col_rand = np.random.randint(1000)
                    color = parent.colormap[col_rand, :3]
                median = parent.add_mask(points=outline, color=color)
                if median is not None:
                    parent.cellcolors = np.append(parent.cellcolors,
                                                  color[np.newaxis, :], axis=0)
                    parent.ncells += 1
        else:
            if dat["masks"].min() == -1:
                dat["masks"] += 1
                dat["outlines"] += 1
            parent.ncells.set(dat["masks"].max())
            if "colors" in dat and len(dat["colors"]) == dat["masks"].max():
                colors = dat["colors"]
            else:
                colors = parent.colormap[:parent.ncells.get(), :3]

            _masks_to_gui(parent, dat["masks"], outlines=dat["outlines"], colors=colors)

            parent.draw_layer()

        if "manual_changes" in dat:
            parent.track_changes = dat["manual_changes"]
            parent.logger.info("loaded in previous changes")
        if "zdraw" in dat:
            parent.zdraw = dat["zdraw"]
        else:
            parent.zdraw = [None for n in range(parent.ncells.get())]
        parent.loaded = True
    else:
        parent.clear_all()

    parent.ismanual = np.zeros(parent.ncells.get(), bool)
    if "ismanual" in dat:
        if len(dat["ismanual"]) == parent.ncells:
            parent.ismanual = dat["ismanual"]

    if "current_channel" in dat:
        # saved as int, loading int is built into .color property
        color = dat['current_channel']
        if isinstance(color, tuple):
            color = color[0]
        parent.color = color

    if "flows" in dat:
        parent.flows = dat["flows"]
        try:
            if parent.flows[0].shape[-3] != dat["masks"].shape[-2]:
                Ly, Lx = dat["masks"].shape[-2:]
                for i in range(len(parent.flows)):
                    parent.flows[i] = cv2.resize(
                        parent.flows[i].squeeze(), (Lx, Ly),
                        interpolation=cv2.INTER_NEAREST)[np.newaxis, ...]
            if parent.NZ == 1:
                parent.recompute_masks = True
            else:
                parent.recompute_masks = False

        except:
            try:
                if len(parent.flows[0]) > 0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[], [], [], [], [[]]]
            parent.recompute_masks = False

    parent.enable_buttons()
    parent.update_layer()
    parent.update_plot()
    del dat
    gc.collect()


def _load_masks(parent, filename=None):
    """ load zeros-based masks (0=no cell, 1=cell 1, ...) """
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load masks (PNG or TIFF)")
        filename = name[0]
    print(f"GUI_INFO: loading masks: {filename}")
    masks = imread(filename)
    outlines = None
    if masks.ndim > 3:
        # Z x nchannels x Ly x Lx
        if masks.shape[-1] > 5:
            parent.flows = list(np.transpose(masks[:, :, :, 2:], (3, 0, 1, 2)))
            outlines = masks[..., 1]
            masks = masks[..., 0]
        else:
            parent.flows = list(np.transpose(masks[:, :, :, 1:], (3, 0, 1, 2)))
            masks = masks[..., 0]
    elif masks.ndim == 3:
        if masks.shape[-1] < 5:
            masks = masks[np.newaxis, :, :, 0]
    elif masks.ndim < 3:
        masks = masks[np.newaxis, :, :]
    # masks should be Z x Ly x Lx
    if masks.shape[0] != parent.NZ:
        print("ERROR: masks are not same depth (number of planes) as image stack")
        return

    _masks_to_gui(parent, masks, outlines)
    if parent.ncells > 0:
        parent.draw_layer()
        parent.toggle_mask_ops()
    del masks
    gc.collect()
    parent.update_layer()


def _masks_to_gui(parent, masks, outlines=None, colors=None):
    """ masks loaded into GUI """
    # get unique values
    shape = masks.shape
    if len(fastremap.unique(masks)) != masks.max() + 1:
        parent.logger.info("renumbering masks")
        fastremap.renumber(masks, in_place=True)
        outlines = None
        masks = masks.reshape(shape)
    if masks.ndim == 2:
        outlines = None
    masks = masks.astype(np.uint16) if masks.max() < 2**16 - 1 else masks.astype(
        np.uint32)
    if parent.restore and "upsample" in parent.restore:
        parent.cellpix_resize = masks.copy()
        parent.cellpix = parent.cellpix_resize.copy()
        parent.cellpix_orig = cv2.resize(
            masks.squeeze(), (parent.Lx0, parent.Ly0),
            interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
        parent.resize = True
    else:
        parent.cellpix = masks
    if parent.cellpix.ndim == 2:
        parent.cellpix = parent.cellpix[np.newaxis, :, :]
        if parent.restore and "upsample" in parent.restore:
            if parent.cellpix_resize.ndim == 2:
                parent.cellpix_resize = parent.cellpix_resize[np.newaxis, :, :]
            if parent.cellpix_orig.ndim == 2:
                parent.cellpix_orig = parent.cellpix_orig[np.newaxis, :, :]

    parent.logger.info(f"{masks.max()} masks found")

    # get outlines
    if outlines is None:  # parent.outlinesOn
        parent.outpix = np.zeros_like(parent.cellpix)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
        for z in range(parent.NZ):
            outlines = masks_to_outlines(parent.cellpix[z])
            parent.outpix[z] = outlines * parent.cellpix[z]
            if parent.restore and "upsample" in parent.restore:
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
            if z % 50 == 0 and parent.NZ > 1:
                parent.logger.info("plane %d outlines processed" % z)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
    else:
        parent.outpix = outlines
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
            for z in range(parent.NZ):
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
                if z % 50 == 0 and parent.NZ > 1:
                    parent.logger.info("plane %d outlines processed" % z)

    if parent.outpix.ndim == 2:
        parent.outpix = parent.outpix[np.newaxis, :, :]
        if parent.restore and "upsample" in parent.restore:
            if parent.outpix_resize.ndim == 2:
                parent.outpix_resize = parent.outpix_resize[np.newaxis, :, :]
            if parent.outpix_orig.ndim == 2:
                parent.outpix_orig = parent.outpix_orig[np.newaxis, :, :]

    parent.ncells.set(parent.cellpix.max())
    colors = parent.colormap[:parent.ncells.get(), :3] if colors is None else colors
    parent.logger.info("creating cellcolors and drawing masks")
    parent.cellcolors = np.concatenate((np.array([[255, 255, 255]]), colors),
                                       axis=0).astype(np.uint8)
    if parent.ncells > 0:
        parent.draw_layer()
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells.get(), bool)
    parent.zdraw = list(-1 * np.ones(parent.ncells.get(), np.int16))

    if hasattr(parent, "stack_filtered"):
        parent.ViewDropDown.setCurrentIndex(parent.ViewDropDown.count() - 1)
        print("set denoised/filtered view")
    else:
        parent.ViewDropDown.setCurrentIndex(0)


def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ == 1:
        if parent.cellpix[0].max() > 65534:
            parent.logger.info("saving 2D masks to tif (too many masks for PNG)")
            imsave(base + "_cp_masks.tif", parent.cellpix[0])
        else:
            parent.logger.info("saving 2D masks to png")
            imsave(base + "_cp_masks.png", parent.cellpix[0].astype(np.uint16))
    else:
        parent.logger.info("saving 3D masks to tiff")
        imsave(base + "_cp_masks.tif", parent.cellpix)


def _save_flows(parent):
    """ save flows and cellprob to tiff """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    parent.logger.info("saving flows and cellprob to tiff")
    if len(parent.flows) > 0:
        imsave(base + "_cp_cellprob.tif", parent.flows[1])
        for i in range(3):
            imsave(base + f"_cp_flows_{i}.tif", parent.flows[0][..., i])
        if len(parent.flows) > 2:
            imsave(base + "_cp_flows.tif", parent.flows[2])
        parent.logger.info("saved flows and cellprob")
    else:
        print("ERROR: no flows or cellprob found")


def _save_rois(parent):
    """ save masks as rois in .zip file for ImageJ """
    filename = parent.filename
    if parent.NZ == 1:
        print(
            f"GUI_INFO: saving {parent.cellpix[0].max()} ImageJ ROIs to .zip archive.")
        save_rois(parent.cellpix[0], parent.filename)
    else:
        print("ERROR: cannot save 3D outlines")


def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ == 1:
        print(
            "GUI_INFO: saving 2D outlines to text file, see docs for info to load into ImageJ"
        )
        outlines = outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        print("ERROR: cannot save 3D outlines")


def _save_sets_with_check(parent):
    """ Save masks and update *_seg.npy file. Use this function when saving should be optional
     based on the disableAutosave checkbox. Otherwise, use _save_sets """
    if not parent.disableAutosave.isChecked():
        _save_sets(parent)


def _save_sets(parent):
    """ save masks to *_seg.npy. This function should be used when saving
    is forced, e.g. when clicking the save button. Otherwise, use _save_sets_with_check
    """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    flow_threshold = parent.segmentation_settings.flow_threshold
    cellprob_threshold = parent.segmentation_settings.cellprob_threshold

    # use ints instead of strings for backwards compatibility with old _seg.npy files
    color_int = list(parent.RGBDropDown.name_map.keys()).index(parent.color)
    if parent.NZ > 1:
        dat = {
            "outlines":
                parent.outpix,
            "colors":
                parent.cellcolors[1:],
            "masks":
                parent.cellpix,
            "current_channel": color_int,
            "filename":
                parent.filename,
            "flows":
                parent.flows,
            "zdraw":
                parent.zdraw,
            "model_path":
                parent.current_model_path
                if hasattr(parent, "current_model_path") else 0,
            "flow_threshold":
                flow_threshold,
            "cellprob_threshold":
                cellprob_threshold,
            "normalize_params":
                parent.get_normalize_params(),
            "restore":
                parent.restore,
            "ratio":
                parent.ratio,
            "diameter":
                parent.segmentation_settings.diameter
        }
        if parent.restore is not None:
            dat["img_restore"] = parent.stack_filtered
    else:
        dat = {
            "outlines":
                parent.outpix.squeeze() if parent.restore is None or
                not "upsample" in parent.restore else parent.outpix_resize.squeeze(),
            "colors":
                parent.cellcolors[1:],
            "masks":
                parent.cellpix.squeeze() if parent.restore is None or
                not "upsample" in parent.restore else parent.cellpix_resize.squeeze(),
            "filename":
                parent.filename,
            "flows":
                parent.flows,
            "ismanual":
                parent.ismanual,
            "manual_changes":
                parent.track_changes,
            "model_path":
                parent.current_model_path
                if hasattr(parent, "current_model_path") else 0,
            "flow_threshold":
                flow_threshold,
            "cellprob_threshold":
                cellprob_threshold,
            "normalize_params":
                parent.get_normalize_params(),
            "restore":
                parent.restore,
            "ratio":
                parent.ratio,
            "diameter":
                parent.segmentation_settings.diameter
        }
        if parent.restore is not None:
            dat["img_restore"] = parent.stack_filtered
    try:
        np.save(base + "_seg.npy", dat)
        parent.logger.info("%d ROIs saved to %s" % (parent.ncells.get(), base + "_seg.npy"))
    except Exception as e:
        print(f"ERROR: {e}")
    del dat
