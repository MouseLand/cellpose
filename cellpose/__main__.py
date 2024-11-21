"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, glob, pathlib, time
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io, version_str, train, denoise
from cellpose.cli import get_arg_parser

try:
    from cellpose.gui import gui3d, gui
    GUI_ENABLED = True
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise

import logging


# settings re-grouped a bit
def main():
    """ Run cellpose from command line
    """

    args = get_arg_parser().parse_args(
    )  # this has to be in a seperate file for autodoc to work

    if args.version:
        print(version_str)
        return

    if args.check_mkl:
        mkl_enabled = models.check_mkl()
    else:
        mkl_enabled = True

    if len(args.dir) == 0 and len(args.image_path) == 0:
        if args.add_model:
            io.add_model(args.add_model)
        else:
            if not GUI_ENABLED:
                print("GUI ERROR: %s" % GUI_ERROR)
                if GUI_IMPORT:
                    print(
                        "GUI FAILED: GUI dependencies may not be installed, to install, run"
                    )
                    print("     pip install 'cellpose[gui]'")
            else:
                if args.Zstack:
                    gui3d.run()
                else:
                    gui.run()

    else:
        if args.verbose:
            from .io import logger_setup
            logger, log_file = logger_setup()
        else:
            print(
                ">>>> !LOGGING OFF BY DEFAULT! To see cellpose progress, set --verbose")
            print("No --verbose => no progress or info printed")
            logger = logging.getLogger(__name__)

        use_gpu = False
        channels = [args.chan, args.chan2]

        # find images
        if len(args.img_filter) > 0:
            imf = args.img_filter
        else:
            imf = None

        # Check with user if they REALLY mean to run without saving anything
        if not (args.train or args.train_size):
            saving_something = args.save_png or args.save_tif or args.save_flows or args.save_txt

        device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu,
                                           device=args.gpu_device)

        if args.pretrained_model is None or args.pretrained_model == "None" or args.pretrained_model == "False" or args.pretrained_model == "0":
            pretrained_model = False
        else:
            pretrained_model = args.pretrained_model

        restore_type = args.restore_type
        if restore_type is not None:
            try:
                denoise.model_path(restore_type)
            except Exception as e:
                raise ValueError("restore_type invalid")
            if args.train or args.train_size:
                raise ValueError("restore_type cannot be used with training on CLI yet")

        if args.transformer and (restore_type is None):
            default_model = "transformer_cp3"
            backbone = "transformer"
        elif args.transformer and restore_type is not None:
            raise ValueError("no transformer based restoration")
        else:
            default_model = "cyto3"
            backbone = "default"

        model_type = None
        if pretrained_model and not os.path.exists(pretrained_model):
            model_type = pretrained_model if pretrained_model is not None else "cyto3"
            model_strings = models.get_user_models()
            all_models = models.MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if ~np.any([model_type == s for s in all_models]):
                model_type = default_model
                logger.warning(
                    f"pretrained model has incorrect path, using {default_model}")
            if model_type == "nuclei":
                szmean = 17.
            else:
                szmean = 30.
        builtin_size = (model_type == "cyto" or model_type == "cyto2" or
                        model_type == "nuclei" or model_type == "cyto3")

        if len(args.image_path) > 0 and (args.train or args.train_size):
            raise ValueError("ERROR: cannot train model with single image input")

        if not args.train and not args.train_size:
            tic = time.time()
            if len(args.dir) > 0:
                image_names = io.get_image_files(
                    args.dir, args.mask_filter, imf=imf,
                    look_one_level_down=args.look_one_level_down)
            else:
                if os.path.exists(args.image_path):
                    image_names = [args.image_path]
                else:
                    raise ValueError(f"ERROR: no file found at {args.image_path}")
            nimg = len(image_names)

            cstr0 = ["GRAY", "RED", "GREEN", "BLUE"]
            cstr1 = ["NONE", "RED", "GREEN", "BLUE"]
            logger.info(
                ">>>> running cellpose on %d images using chan_to_seg %s and chan (opt) %s"
                % (nimg, cstr0[channels[0]], cstr1[channels[1]]))

            # handle built-in model exceptions
            if builtin_size and restore_type is None and not args.pretrained_model_ortho:
                model = models.Cellpose(gpu=gpu, device=device, model_type=model_type,
                                        backbone=backbone)
            else:
                builtin_size = False
                if args.all_channels:
                    channels = None
                    img = io.imread(image_names[0])
                    if img.ndim == 3:
                        nchan = min(img.shape)
                    elif img.ndim == 2:
                        nchan = 1
                    channels = None
                else:
                    nchan = 2

                pretrained_model = None if model_type is not None else pretrained_model
                if restore_type is None:
                    pretrained_model_ortho = None if args.pretrained_model_ortho is None else args.pretrained_model_ortho
                    model = models.CellposeModel(gpu=gpu, device=device,
                                                 pretrained_model=pretrained_model,
                                                 model_type=model_type,
                                                 nchan=nchan,
                                                 backbone=backbone,
                                                 pretrained_model_ortho=pretrained_model_ortho)
                else:
                    model = denoise.CellposeDenoiseModel(
                        gpu=gpu, device=device, pretrained_model=pretrained_model,
                        model_type=model_type, restore_type=restore_type, nchan=nchan,
                        chan2_restore=args.chan2_restore)

            # handle diameters
            if args.diameter == 0:
                if builtin_size:
                    diameter = None
                    logger.info(">>>> estimating diameter for each image")
                else:
                    if restore_type is None:
                        logger.info(
                            ">>>> not using cyto3, cyto, cyto2, or nuclei model, cannot auto-estimate diameter"
                        )
                    else:
                        logger.info(
                            ">>>> cannot auto-estimate diameter for image restoration")
                    diameter = model.diam_labels
                    logger.info(">>>> using diameter %0.3f for all images" % diameter)
            else:
                diameter = args.diameter
                logger.info(">>>> using diameter %0.3f for all images" % diameter)

            tqdm_out = utils.TqdmToLogger(logger, level=logging.INFO)

            for image_name in tqdm(image_names, file=tqdm_out):
                image = io.imread(image_name)
                out = model.eval(
                    image, channels=channels, diameter=diameter, do_3D=args.do_3D,
                    augment=args.augment, resample=(not args.no_resample),
                    flow_threshold=args.flow_threshold,
                    cellprob_threshold=args.cellprob_threshold,
                    stitch_threshold=args.stitch_threshold, min_size=args.min_size,
                    invert=args.invert, batch_size=args.batch_size,
                    interp=(not args.no_interp), normalize=(not args.no_norm),
                    channel_axis=args.channel_axis, z_axis=args.z_axis,
                    anisotropy=args.anisotropy, niter=args.niter,
                    dP_smooth=args.dP_smooth)
                masks, flows = out[:2]
                if len(out) > 3 and restore_type is None:
                    diams = out[-1]
                else:
                    diams = diameter
                ratio = 1.
                if restore_type is not None:
                    imgs_dn = out[-1]
                    ratio = diams / model.dn.diam_mean if "upsample" in restore_type else 1.
                    diams = model.dn.diam_mean if "upsample" in restore_type and model.dn.diam_mean > diams else diams
                else:
                    imgs_dn = None
                if args.exclude_on_edges:
                    masks = utils.remove_edge_masks(masks)
                if not args.no_npy:
                    io.masks_flows_to_seg(image, masks, flows, image_name,
                                          imgs_restore=imgs_dn, channels=channels,
                                          diams=diams, restore_type=restore_type,
                                          ratio=1.)
                if saving_something:
                    io.save_masks(image, masks, flows, image_name, png=args.save_png,
                                  tif=args.save_tif, save_flows=args.save_flows,
                                  save_outlines=args.save_outlines,
                                  dir_above=args.dir_above, savedir=args.savedir,
                                  save_txt=args.save_txt, in_folders=args.in_folders,
                                  save_mpl=args.save_mpl)
                if args.save_rois:
                    io.save_rois(masks, image_name)
            logger.info(">>>> completed in %0.3f sec" % (time.time() - tic))
        else:
            test_dir = None if len(args.test_dir) == 0 else args.test_dir
            images, labels, image_names, train_probs = None, None, None, None
            test_images, test_labels, image_names_test, test_probs = None, None, None, None
            compute_flows = False
            if len(args.file_list) > 0:
                if os.path.exists(args.file_list):
                    dat = np.load(args.file_list, allow_pickle=True).item()
                    image_names = dat["train_files"]
                    image_names_test = dat.get("test_files", None)
                    train_probs = dat.get("train_probs", None)
                    test_probs = dat.get("test_probs", None)
                    compute_flows = dat.get("compute_flows", False)
                    load_files = False
                else:
                    logger.critical(f"ERROR: {args.file_list} does not exist")
            else:
                output = io.load_train_test_data(args.dir, test_dir, imf,
                                                 args.mask_filter,
                                                 args.look_one_level_down)
                images, labels, image_names, test_images, test_labels, image_names_test = output
                load_files = True

            # training with all channels
            if args.all_channels:
                img = images[0] if images is not None else io.imread(image_names[0])
                if img.ndim == 3:
                    nchan = min(img.shape)
                elif img.ndim == 2:
                    nchan = 1
                channels = None
            else:
                nchan = 2

            # model path
            szmean = args.diam_mean
            if not os.path.exists(pretrained_model) and model_type is None:
                if not args.train:
                    error_message = "ERROR: model path missing or incorrect - cannot train size model"
                    logger.critical(error_message)
                    raise ValueError(error_message)
                pretrained_model = False
                logger.info(">>>> training from scratch")
            if args.train:
                logger.info(
                    ">>>> during training rescaling images to fixed diameter of %0.1f pixels"
                    % args.diam_mean)

            # initialize model
            model = models.CellposeModel(
                device=device, model_type=model_type, diam_mean=szmean, nchan=nchan,
                pretrained_model=pretrained_model if model_type is None else None,
                backbone=backbone)

            # train segmentation model
            if args.train:
                cpmodel_path = train.train_seg(
                    model.net, images, labels, train_files=image_names,
                    test_data=test_images, test_labels=test_labels,
                    test_files=image_names_test, train_probs=train_probs,
                    test_probs=test_probs, compute_flows=compute_flows,
                    load_files=load_files, normalize=(not args.no_norm),
                    channels=channels, channel_axis=args.channel_axis, rgb=(nchan == 3),
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    SGD=args.SGD, n_epochs=args.n_epochs, batch_size=args.batch_size,
                    min_train_masks=args.min_train_masks,
                    nimg_per_epoch=args.nimg_per_epoch,
                    nimg_test_per_epoch=args.nimg_test_per_epoch,
                    save_path=os.path.realpath(args.dir), save_every=args.save_every,
                    model_name=args.model_name_out)[0]
                model.pretrained_model = cpmodel_path
                logger.info(">>>> model trained and saved to %s" % cpmodel_path)

            # train size model
            if args.train_size:
                sz_model = models.SizeModel(cp_model=model, device=device)
                # data has already been normalized and reshaped
                sz_model.params = train.train_size(
                    model.net, model.pretrained_model, images, labels,
                    train_files=image_names, test_data=test_images,
                    test_labels=test_labels, test_files=image_names_test,
                    train_probs=train_probs, test_probs=test_probs,
                    load_files=load_files, channels=channels,
                    min_train_masks=args.min_train_masks,
                    channel_axis=args.channel_axis, rgb=(nchan == 3),
                    nimg_per_epoch=args.nimg_per_epoch, normalize=(not args.no_norm),
                    nimg_test_per_epoch=args.nimg_test_per_epoch,
                    batch_size=args.batch_size)
                if test_images is not None:
                    test_masks = [lbl[0] for lbl in test_labels
                                 ] if test_labels is not None else test_labels
                    predicted_diams, diams_style = sz_model.eval(
                        test_images, channels=channels)
                    ccs = np.corrcoef(
                        diams_style,
                        np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0, 1]
                    cc = np.corrcoef(
                        predicted_diams,
                        np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0, 1]
                    logger.info(
                        "style test correlation: %0.4f; final test correlation: %0.4f" %
                        (ccs, cc))
                    np.save(
                        os.path.join(
                            args.test_dir,
                            "%s_predicted_diams.npy" % os.path.split(cpmodel_path)[1]),
                        {
                            "predicted_diams": predicted_diams,
                            "diams_style": diams_style
                        })


if __name__ == "__main__":
    main()
