"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, datetime, gc, warnings, glob, shutil
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from tqdm import tqdm
from pathlib import Path
import re
from . import version_str
from roifile import ImagejRoi, roiwrite

try:
    from qtpy import QtGui, QtCore, Qt, QtWidgets
    from qtpy.QtWidgets import QMessageBox
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    import nd2
    ND2 = True
except:
    ND2 = False

try:
    import nrrd
    NRRD = True
except:
    NRRD = False

try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

io_logger = logging.getLogger(__name__)


def logger_setup():
    cp_dir = pathlib.Path.home().joinpath(".cellpose")
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath("run.log")
    try:
        log_file.unlink()
    except:
        print("creating new log file")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    logger.info(f"WRITING LOG OUTPUT TO {log_file}")
    logger.info(version_str)
    #logger.handlers[1].stream = sys.stdout

    return logger, log_file


from . import utils, plot, transforms


# helper function to check for a path; if it doesn"t exist, make it
def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def outlines_to_text(base, outlines):
    with open(base + "_cp_outlines.txt", "w") as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ",".join(map(str, xy))
            f.write(xy_str)
            f.write("\n")


def load_dax(filename):
    ### modified from ZhuangLab github:
    ### https://github.com/ZhuangLab/storm-analysis/blob/71ae493cbd17ddb97938d0ae2032d97a0eaa76b2/storm_analysis/sa_library/datareader.py#L156

    inf_filename = os.path.splitext(filename)[0] + ".inf"
    if not os.path.exists(inf_filename):
        io_logger.critical(
            f"ERROR: no inf file found for dax file {filename}, cannot load dax without it"
        )
        return None

    ### get metadata
    image_height, image_width = None, None
    # extract the movie information from the associated inf file
    size_re = re.compile(r"frame dimensions = ([\d]+) x ([\d]+)")
    length_re = re.compile(r"number of frames = ([\d]+)")
    endian_re = re.compile(r" (big|little) endian")

    with open(inf_filename, "r") as inf_file:
        lines = inf_file.read().split("\n")
        for line in lines:
            m = size_re.match(line)
            if m:
                image_height = int(m.group(2))
                image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    bigendian = 1
                else:
                    bigendian = 0
    # set defaults, warn the user that they couldn"t be determined from the inf file.
    if not image_height:
        io_logger.warning("could not determine dax image size, assuming 256x256")
        image_height = 256
        image_width = 256

    ### load image
    img = np.memmap(filename, dtype="uint16",
                    shape=(number_frames, image_height, image_width))
    if bigendian:
        img = img.byteswap()
    img = np.array(img)

    return img


def imread(filename):
    """
    Read in an image file with tif or image file type supported by cv2.

    Args:
        filename (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data as a NumPy array.

    Raises:
        None

    Raises an error if the image file format is not supported.

    Examples:
        >>> img = imread("image.tif")
    """
    # ensure that extension check is not case sensitive
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff":
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]["shape"]
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f"reading tiff with {ltif} planes")
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i, page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)
        return img
    elif ext == ".dax":
        img = load_dax(filename)
        return img
    elif ext == ".nd2":
        if not ND2:
            io_logger.critical("ERROR: need to 'pip install nd2' to load in .nd2 file")
            return None
    elif ext == ".nrrd":
        if not NRRD:
            io_logger.critical(
                "ERROR: need to 'pip install pynrrd' to load in .nrrd file")
            return None
        else:
            img, metadata = nrrd.read(filename)
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            return img
    elif ext != ".npy":
        try:
            img = cv2.imread(filename, -1)  #cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2, 1, 0]]
            return img
        except Exception as e:
            io_logger.critical("ERROR: could not read file, %s" % e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat["masks"]
            return masks
        except Exception as e:
            io_logger.critical("ERROR: could not read masks from file, %s" % e)
            return None


def remove_model(filename, delete=False):
    """ remove model from .cellpose custom model list """
    filename = os.path.split(filename)[-1]
    from . import models
    model_strings = models.get_user_models()
    if len(model_strings) > 0:
        with open(models.MODEL_LIST_PATH, "w") as textfile:
            for fname in model_strings:
                textfile.write(fname + "\n")
    else:
        # write empty file
        textfile = open(models.MODEL_LIST_PATH, "w")
        textfile.close()
    print(f"{filename} removed from custom model list")
    if delete:
        os.remove(os.fspath(models.MODEL_DIR.joinpath(fname)))
        print("model deleted")


def add_model(filename):
    """ add model to .cellpose models folder to use with GUI or CLI """
    from . import models
    fname = os.path.split(filename)[-1]
    try:
        shutil.copyfile(filename, os.fspath(models.MODEL_DIR.joinpath(fname)))
    except shutil.SameFileError:
        pass
    print(f"{filename} copied to models folder {os.fspath(models.MODEL_DIR)}")
    if fname not in models.get_user_models():
        with open(models.MODEL_LIST_PATH, "a") as textfile:
            textfile.write(fname + "\n")


def imsave(filename, arr):
    """
    Saves an image array to a file.

    Args:
        filename (str): The name of the file to save the image to.
        arr (numpy.ndarray): The image array to be saved.

    Returns:
        None
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff":
        tifffile.imwrite(filename, arr)
    else:
        if len(arr.shape) > 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)


def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """
    Finds all images in a folder and its subfolders (if specified) with the given file extensions.

    Args:
        folder (str): The path to the folder to search for images.
        mask_filter (str): The filter for mask files.
        imf (str, optional): The additional filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to search for images in subfolders. Defaults to False.

    Returns:
        list: A list of image file paths.

    Raises:
        ValueError: If no files are found in the specified folder.
        ValueError: If no images are found in the specified folder with the supported file extensions.
        ValueError: If no images are found in the specified folder without the mask or flow file endings.
    """
    mask_filters = ["_cp_masks", "_cp_output", "_flows", "_masks", mask_filter]
    image_names = []
    if imf is None:
        imf = ""

    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dax", ".nd2", ".nrrd"]
    l0 = 0
    al = 0
    for folder in folders:
        all_files = glob.glob(folder + "/*")
        al += len(all_files)
        for ext in exts:
            image_names.extend(glob.glob(folder + f"/*{imf}{ext}"))
            image_names.extend(glob.glob(folder + f"/*{imf}{ext.upper()}"))
        l0 += len(image_names)

    # return error if no files found
    if al == 0:
        raise ValueError("ERROR: no files in --dir folder ")
    elif l0 == 0:
        raise ValueError(
            "ERROR: no images in --dir folder with extensions .png, .jpg, .jpeg, .tif, .tiff"
        )

    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and
                      imfile[-len(mask_filter):] != mask_filter) or
                     len(imfile) <= len(mask_filter) for mask_filter in mask_filters])
        if len(imf) > 0:
            igood &= imfile[-len(imf):] == imf
        if igood:
            imn.append(im)

    image_names = imn

    # remove duplicates
    image_names = [*set(image_names)]
    image_names = natsorted(image_names)

    if len(image_names) == 0:
        raise ValueError(
            "ERROR: no images in --dir folder without _masks or _flows ending")

    return image_names


def get_label_files(image_names, mask_filter, imf=None):
    """
    Get the label files corresponding to the given image names and mask filter.

    Args:
        image_names (list): List of image names.
        mask_filter (str): Mask filter to be applied.
        imf (str, optional): Image file extension. Defaults to None.

    Returns:
        tuple: A tuple containing the label file names and flow file names (if present).
    """
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    # check for flows
    if os.path.exists(label_names0[0] + "_flows.tif"):
        flow_names = [label_names0[n] + "_flows.tif" for n in range(nimg)]
    else:
        flow_names = [label_names[n] + "_flows.tif" for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        io_logger.info(
            "not all flows are present, running flow generation for all images")
        flow_names = None

    # check for masks
    if mask_filter == "_seg.npy":
        label_names = [label_names[n] + mask_filter for n in range(nimg)]
        return label_names, None

    if os.path.exists(label_names[0] + mask_filter + ".tif"):
        label_names = [label_names[n] + mask_filter + ".tif" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".tiff"):
        label_names = [label_names[n] + mask_filter + ".tiff" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".png"):
        label_names = [label_names[n] + mask_filter + ".png" for n in range(nimg)]
    # todo, allow _seg.npy
    #elif os.path.exists(label_names[0] + "_seg.npy"):
    #    io_logger.info("labels found as _seg.npy files, converting to tif")
    else:
        if not flow_names:
            raise ValueError("labels not provided with correct --mask_filter")
        else:
            label_names = None
    if not all([os.path.exists(label) for label in label_names]):
        if not flow_names:
            raise ValueError(
                "labels not provided for all images in train and/or test set")
        else:
            label_names = None

    return label_names, flow_names


def load_images_labels(tdir, mask_filter="_masks", image_filter=None,
                       look_one_level_down=False):
    """
    Loads images and corresponding labels from a directory.

    Args:
        tdir (str): The directory path.
        mask_filter (str, optional): The filter for mask files. Defaults to "_masks".
        image_filter (str, optional): The filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to look for files one level down. Defaults to False.

    Returns:
        tuple: A tuple containing a list of images, a list of labels, and a list of image names.
    """
    image_names = get_image_files(tdir, mask_filter, image_filter, look_one_level_down)
    nimg = len(image_names)

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter,
                                              imf=image_filter)

    images = []
    labels = []
    k = 0
    for n in range(nimg):
        if (os.path.isfile(label_names[n]) or 
            (flow_names is not None and os.path.isfile(flow_names[0]))):
            image = imread(image_names[n])
            if label_names is not None:
                label = imread(label_names[n])
            if flow_names is not None:
                flow = imread(flow_names[n])
                if flow.shape[0] < 4:
                    label = np.concatenate((label[np.newaxis, :, :], flow), axis=0)
                else:
                    label = flow
            images.append(image)
            labels.append(label)
            k += 1
    io_logger.info(f"{k} / {nimg} images in {tdir} folder have labels")
    return images, labels, image_names


def load_train_test_data(train_dir, test_dir=None, image_filter=None,
                         mask_filter="_masks", look_one_level_down=False):
    """
    Loads training and testing data for a Cellpose model.

    Args:
        train_dir (str): The directory path containing the training data.
        test_dir (str, optional): The directory path containing the testing data. Defaults to None.
        image_filter (str, optional): The filter for selecting image files. Defaults to None.
        mask_filter (str, optional): The filter for selecting mask files. Defaults to "_masks".
        look_one_level_down (bool, optional): Whether to look for data in subdirectories of train_dir and test_dir. Defaults to False.

    Returns:
        images (list): A list of training images.
        labels (list): A list of labels corresponding to the training images.
        image_names (list): A list of names of the training images.
        test_images (list, optional): A list of testing images. None if test_dir is not provided.
        test_labels (list, optional): A list of labels corresponding to the testing images. None if test_dir is not provided.
        test_image_names (list, optional): A list of names of the testing images. None if test_dir is not provided.
    """
    images, labels, image_names = load_images_labels(train_dir, mask_filter,
                                                     image_filter, look_one_level_down)

    # testing data
    test_images, test_labels, test_image_names = None, None, None
    if test_dir is not None:
        test_images, test_labels, test_image_names = load_images_labels(
            test_dir, mask_filter, image_filter, look_one_level_down)

    return images, labels, image_names, test_images, test_labels, test_image_names


def masks_flows_to_seg(images, masks, flows, file_names, diams=30., channels=None,
                       imgs_restore=None, restore_type=None, ratio=1.):
    """Save output of model eval to be loaded in GUI.

    Can be list output (run on multiple images) or single output (run on single image).

    Saved to file_names[k]+"_seg.npy".
    
    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        diams (float array): Diameters used to run Cellpose. Defaults to 30.
        channels (list, int, optional): Channels used to run Cellpose. Defaults to None.

    Returns:
        None
    """

    if channels is None:
        channels = [0, 0]

    if isinstance(masks, list):
        if not isinstance(diams, (list, np.ndarray)):
            diams = diams * np.ones(len(masks), np.float32)
        if imgs_restore is None:
            imgs_restore = [None] * len(masks)
        if isinstance(file_names, str):
            file_names = [file_names] * len(masks)
        for k, [image, mask, flow, diam, file_name, img_restore
               ] in enumerate(zip(images, masks, flows, diams, file_names,
                                  imgs_restore)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, file_name, diams=diam,
                               channels=channels_img, imgs_restore=img_restore,
                               restore_type=restore_type, ratio=ratio)
        return

    if len(channels) == 1:
        channels = channels[0]

    flowi = []
    if flows[0].ndim == 3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(
            cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,
                                                                            ...])
    else:
        flowi.append(flows[0])

    if flows[0].ndim == 3:
        cellprob = (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(
            np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis, ...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis, ...]
    else:
        flowi.append(
            (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(np.uint8))
        flowi.append((flows[1][0] / 10 * 127 + 127).astype(np.uint8))
    if len(flows) > 2:
        flowi.append(flows[3])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis, ...]), axis=0))
    outlines = masks * utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]

    dat = {
        "outlines":
            outlines.astype(np.uint16) if outlines.max() < 2**16 -
            1 else outlines.astype(np.uint32),
        "masks":
            masks.astype(np.uint16) if outlines.max() < 2**16 -
            1 else masks.astype(np.uint32),
        "chan_choose":
            channels,
        "ismanual":
            np.zeros(masks.max(), bool),
        "filename":
            file_names,
        "flows":
            flowi,
        "diameter":
            diams
    }
    if restore_type is not None and imgs_restore is not None:
        dat["restore"] = restore_type
        dat["ratio"] = ratio
        dat["img_restore"] = imgs_restore

    np.save(base + "_seg.npy", dat)


def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True) 
    
        does not work for 3D images
    
    """
    save_masks(images, masks, flows, file_names, png=True)


def save_rois(masks, file_name):
    """ save masks to .roi files in .zip archive for ImageJ/Fiji

    Args:
        masks (np.ndarray): masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels
        file_name (str): name to save the .zip file to
    
    Returns:
        None
    """
    outlines = utils.outlines_list(masks)
    rois = [ImagejRoi.frompoints(outline) for outline in outlines]
    file_name = os.path.splitext(file_name)[0] + "_rois.zip"

    # Delete file if it exists; the roifile lib appends to existing zip files.
    # If the user removed a mask it will still be in the zip file
    if os.path.exists(file_name):
        os.remove(file_name)

    roiwrite(file_name, rois)


def save_masks(images, masks, flows, file_names, png=True, tif=False, channels=[0, 0],
               suffix="", save_flows=False, save_outlines=False, dir_above=False,
               in_folders=False, savedir=None, save_txt=False, save_mpl=False):
    """ Save masks + nicely plotted segmentation image to png and/or tiff.

    Can save masks, flows to different directories, if in_folders is True.

    If png, masks[k] for images[k] are saved to file_names[k]+"_cp_masks.png".

    If tif, masks[k] for images[k] are saved to file_names[k]+"_cp_masks.tif".

    If png and matplotlib installed, full segmentation figure is saved to file_names[k]+"_cp.png".

    Only tif option works for 3D data, and only tif option works for empty masks.

    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        png (bool, optional): Save masks to PNG. Defaults to True.
        tif (bool, optional): Save masks to TIF. Defaults to False.
        channels (list, int, optional): Channels used to run Cellpose. Defaults to [0,0].
        suffix (str, optional): Add name to saved masks. Defaults to "".
        save_flows (bool, optional): Save flows output from Cellpose.eval. Defaults to False.
        save_outlines (bool, optional): Save outlines of masks. Defaults to False.
        dir_above (bool, optional): Save masks/flows in directory above. Defaults to False.
        in_folders (bool, optional): Save masks/flows in separate folders. Defaults to False.
        savedir (str, optional): Absolute path where images will be saved. If None, saves to image directory. Defaults to None.
        save_txt (bool, optional): Save masks as list of outlines for ImageJ. Defaults to False.
        save_mpl (bool, optional): If True, saves a matplotlib figure of the original image/segmentation/flows. Does not work for 3D.
                This takes a long time for large images. Defaults to False.
    
    Returns:
        None
    """

    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif, suffix=suffix,
                       dir_above=dir_above, save_flows=save_flows,
                       save_outlines=save_outlines, savedir=savedir, save_txt=save_txt,
                       in_folders=in_folders, save_mpl=save_mpl)
        return

    if masks.ndim > 2 and not tif:
        raise ValueError("cannot save 3D outputs as PNG, use tif option instead")

    if masks.max() == 0:
        io_logger.warning("no masks found, will not save PNG or outlines")
        if not tif:
            return
        else:
            png = False
            save_outlines = False
            save_flows = False
            save_txt = False

    if savedir is None:
        if dir_above:
            savedir = Path(file_names).parent.parent.absolute(
            )  #go up a level to save in its own folder
        else:
            savedir = Path(file_names).parent.absolute()

    check_dir(savedir)

    basename = os.path.splitext(os.path.basename(file_names))[0]
    if in_folders:
        maskdir = os.path.join(savedir, "masks")
        outlinedir = os.path.join(savedir, "outlines")
        txtdir = os.path.join(savedir, "txt_outlines")
        flowdir = os.path.join(savedir, "flows")
    else:
        maskdir = savedir
        outlinedir = savedir
        txtdir = savedir
        flowdir = savedir

    check_dir(maskdir)

    exts = []
    if masks.ndim > 2:
        png = False
        tif = True
    if png:
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16)
            exts.append(".png")
        else:
            png = False
            tif = True
            io_logger.warning(
                "found more than 65535 masks in each image, cannot save PNG, saving as TIF"
            )
    if tif:
        exts.append(".tif")

    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:

            imsave(os.path.join(maskdir, basename + "_cp_masks" + suffix + ext), masks)

    if save_mpl and png and MATPLOTLIB and not min(images.shape) > 3:
        # Make and save original/segmentation/flows image

        img = images.copy()
        if img.ndim < 3:
            img = img[:, :, np.newaxis]
        elif img.shape[0] < 8:
            np.transpose(img, (1, 2, 0))

        fig = plt.figure(figsize=(12, 3))
        plot.show_segmentation(fig, img, masks, flows[0])
        fig.savefig(os.path.join(savedir, basename + "_cp_output" + suffix + ".png"),
                    dpi=300)
        plt.close(fig)

    # ImageJ txt outline files
    if masks.ndim < 3 and save_txt:
        check_dir(txtdir)
        outlines = utils.outlines_list(masks)
        outlines_to_text(os.path.join(txtdir, basename), outlines)

    # RGB outline images
    if masks.ndim < 3 and save_outlines:
        check_dir(outlinedir)
        outlines = utils.masks_to_outlines(masks)
        outX, outY = np.nonzero(outlines)
        img0 = transforms.normalize99(images)
        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1, 2, 0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=channels)
        else:
            if img0.max() <= 50.0:
                img0 = np.uint8(np.clip(img0 * 255, 0, 1))
        imgout = img0.copy()
        imgout[outX, outY] = np.array([255, 0, 0])  #pure red
        imsave(os.path.join(outlinedir, basename + "_outlines" + suffix + ".png"),
               imgout)

    # save RGB flow picture
    if masks.ndim < 3 and save_flows:
        check_dir(flowdir)
        imsave(os.path.join(flowdir, basename + "_flows" + suffix + ".tif"),
               (flows[0] * (2**16 - 1)).astype(np.uint16))
        #save full flow data
        imsave(os.path.join(flowdir, basename + "_dP" + suffix + ".tif"), flows[1])
