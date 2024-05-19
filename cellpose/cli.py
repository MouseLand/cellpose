"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu and Michael Rariden.
"""

import argparse


def get_arg_parser():
    """ Parses command line arguments for cellpose main function

    Note: this function has to be in a separate file to allow autodoc to work for CLI.
    The autodoc_mock_imports in conf.py does not work for sphinx-argparse sometimes,
    see https://github.com/ashb/sphinx-argparse/issues/9#issue-1097057823
    """

    parser = argparse.ArgumentParser(description="Cellpose Command Line Parameters")

    # misc settings
    parser.add_argument("--version", action="store_true",
                        help="show cellpose version info")
    parser.add_argument(
        "--verbose", action="store_true",
        help="show information about running and settings and save to log")
    parser.add_argument("--Zstack", action="store_true", help="run GUI in 3D mode")

    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("Hardware Arguments")
    hardware_args.add_argument("--use_gpu", action="store_true",
                               help="use gpu if torch with cuda installed")
    hardware_args.add_argument(
        "--gpu_device", required=False, default="0", type=str,
        help="which gpu device to use, use an integer for torch, or mps for M1")
    hardware_args.add_argument("--check_mkl", action="store_true",
                               help="check if mkl working")

    # settings for locating and formatting images
    input_img_args = parser.add_argument_group("Input Image Arguments")
    input_img_args.add_argument("--dir", default=[], type=str,
                                help="folder containing data to run or train on.")
    input_img_args.add_argument(
        "--image_path", default=[], type=str, help=
        "if given and --dir not given, run on single image instead of folder (cannot train with this option)"
    )
    input_img_args.add_argument(
        "--look_one_level_down", action="store_true",
        help="run processing on all subdirectories of current folder")
    input_img_args.add_argument("--img_filter", default=[], type=str,
                                help="end string for images to run on")
    input_img_args.add_argument(
        "--channel_axis", default=None, type=int,
        help="axis of image which corresponds to image channels")
    input_img_args.add_argument("--z_axis", default=None, type=int,
                                help="axis of image which corresponds to Z dimension")
    input_img_args.add_argument(
        "--chan", default=0, type=int, help=
        "channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s")
    input_img_args.add_argument(
        "--chan2", default=0, type=int, help=
        "nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s"
    )
    input_img_args.add_argument("--invert", action="store_true",
                                help="invert grayscale channel")
    input_img_args.add_argument(
        "--all_channels", action="store_true", help=
        "use all channels in image if using own model and images with special channels")

    # model settings
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--pretrained_model", required=False, default="cyto",
                            type=str,
                            help="model to use for running or starting training")
    model_args.add_argument("--restore_type", required=False, default=None, type=str,
                            help="model to use for image restoration")
    model_args.add_argument("--chan2_restore", action="store_true",
                            help="use nuclei restore model for second channel")
    model_args.add_argument(
        "--add_model", required=False, default=None, type=str,
        help="model path to copy model to hidden .cellpose folder for using in GUI/CLI")
    model_args.add_argument(
        "--transformer", action="store_true", help=
        "use transformer backbone (pretrained_model from Cellpose3 is transformer_cp3)")

    # algorithm settings
    algorithm_args = parser.add_argument_group("Algorithm Arguments")
    algorithm_args.add_argument(
        "--no_resample", action="store_true", help=
        "disable dynamics on full image (makes algorithm faster for images with large diameters)"
    )
    algorithm_args.add_argument(
        "--no_interp", action="store_true",
        help="do not interpolate when running dynamics (was default)")
    algorithm_args.add_argument("--no_norm", action="store_true",
                                help="do not normalize images (normalize=False)")
    algorithm_args.add_argument(
        "--do_3D", action="store_true",
        help="process images as 3D stacks of images (nplanes x nchan x Ly x Lx")
    algorithm_args.add_argument(
        "--diameter", required=False, default=30., type=float, help=
        "cell diameter, if 0 will use the diameter of the training labels used in the model, or with built-in model will estimate diameter for each image"
    )
    algorithm_args.add_argument(
        "--stitch_threshold", required=False, default=0.0, type=float,
        help="compute masks in 2D then stitch together masks with IoU>0.9 across planes"
    )
    algorithm_args.add_argument(
        "--min_size", required=False, default=15, type=int,
        help="minimum number of pixels per mask, can turn off with -1")

    algorithm_args.add_argument(
        "--flow_threshold", default=0.4, type=float, help=
        "flow error threshold, 0 turns off this optional QC step. Default: %(default)s")
    algorithm_args.add_argument(
        "--cellprob_threshold", default=0, type=float,
        help="cellprob threshold, default is 0, decrease to find more and larger masks")
    algorithm_args.add_argument(
        "--niter", default=0, type=int, help=
        "niter, number of iterations for dynamics for mask creation, default of 0 means it is proportional to diameter, set to a larger number like 2000 for very long ROIs"
    )

    algorithm_args.add_argument("--anisotropy", required=False, default=1.0, type=float,
                                help="anisotropy of volume in 3D")
    algorithm_args.add_argument("--exclude_on_edges", action="store_true",
                                help="discard masks which touch edges of image")
    algorithm_args.add_argument(
        "--augment", action="store_true",
        help="tiles image with overlapping tiles and flips overlapped regions to augment"
    )

    # output settings
    output_args = parser.add_argument_group("Output Arguments")
    output_args.add_argument(
        "--save_png", action="store_true",
        help="save masks as png and outlines as text file for ImageJ")
    output_args.add_argument(
        "--save_tif", action="store_true",
        help="save masks as tif and outlines as text file for ImageJ")
    output_args.add_argument("--no_npy", action="store_true",
                             help="suppress saving of npy")
    output_args.add_argument(
        "--savedir", default=None, type=str, help=
        "folder to which segmentation results will be saved (defaults to input image directory)"
    )
    output_args.add_argument(
        "--dir_above", action="store_true", help=
        "save output folders adjacent to image folder instead of inside it (off by default)"
    )
    output_args.add_argument("--in_folders", action="store_true",
                             help="flag to save output in folders (off by default)")
    output_args.add_argument(
        "--save_flows", action="store_true", help=
        "whether or not to save RGB images of flows when masks are saved (disabled by default)"
    )
    output_args.add_argument(
        "--save_outlines", action="store_true", help=
        "whether or not to save RGB outline images when masks are saved (disabled by default)"
    )
    output_args.add_argument(
        "--save_rois", action="store_true",
        help="whether or not to save ImageJ compatible ROI archive (disabled by default)"
    )
    output_args.add_argument(
        "--save_txt", action="store_true",
        help="flag to enable txt outlines for ImageJ (disabled by default)")
    output_args.add_argument(
        "--save_mpl", action="store_true",
        help="save a figure of image/mask/flows using matplotlib (disabled by default). "
        "This is slow, especially with large images.")

    # training settings
    training_args = parser.add_argument_group("Training Arguments")
    training_args.add_argument("--train", action="store_true",
                               help="train network using images in dir")
    training_args.add_argument("--train_size", action="store_true",
                               help="train size network at end of training")
    training_args.add_argument("--test_dir", default=[], type=str,
                               help="folder containing test data (optional)")
    training_args.add_argument(
        "--file_list", default=[], type=str, help=
        "path to list of files for training and testing and probabilities for each image (optional)"
    )
    training_args.add_argument(
        "--mask_filter", default="_masks", type=str, help=
        "end string for masks to run on. use '_seg.npy' for manual annotations from the GUI. Default: %(default)s"
    )
    training_args.add_argument(
        "--diam_mean", default=30., type=float, help=
        "mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0"
    )
    training_args.add_argument("--learning_rate", default=0.2, type=float,
                               help="learning rate. Default: %(default)s")
    training_args.add_argument("--weight_decay", default=0.00001, type=float,
                               help="weight decay. Default: %(default)s")
    training_args.add_argument("--n_epochs", default=500, type=int,
                               help="number of epochs. Default: %(default)s")
    training_args.add_argument("--batch_size", default=8, type=int,
                               help="batch size. Default: %(default)s")
    training_args.add_argument(
        "--nimg_per_epoch", default=None, type=int,
        help="number of train images per epoch. Default is to use all train images.")
    training_args.add_argument(
        "--nimg_test_per_epoch", default=None, type=int,
        help="number of test images per epoch. Default is to use all test images.")
    training_args.add_argument(
        "--min_train_masks", default=5, type=int, help=
        "minimum number of masks a training image must have to be used. Default: %(default)s"
    )
    training_args.add_argument("--SGD", default=1, type=int, help="use SGD")
    training_args.add_argument(
        "--save_every", default=100, type=int,
        help="number of epochs to skip between saves. Default: %(default)s")
    training_args.add_argument(
        "--model_name_out", default=None, type=str,
        help="Name of model to save as, defaults to name describing model architecture. "
        "Model is saved in the folder specified by --dir in models subfolder.")

    return parser
