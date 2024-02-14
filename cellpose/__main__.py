"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, glob, pathlib, time
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io, version_str, train
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

    args = get_arg_parser().parse_args()  # this has to be in a seperate file for autodoc to work

    if args.version:
        print(version_str)
        return

    if args.check_mkl:
        mkl_enabled = models.check_mkl()
    else:
        mkl_enabled = True
    
    if len(args.dir)==0 and len(args.image_path)==0:
        if args.add_model:
            io.add_model(args.add_model)
        else:
            if not GUI_ENABLED:
                print('GUI ERROR: %s'%GUI_ERROR)
                if GUI_IMPORT:
                    print('GUI FAILED: GUI dependencies may not be installed, to install, run')
                    print('     pip install "cellpose[gui]"')
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
            print('>>>> !LOGGING OFF BY DEFAULT! To see cellpose progress, set --verbose')
            print('No --verbose => no progress or info printed')
            logger = logging.getLogger(__name__)

        use_gpu = False
        channels = [args.chan, args.chan2]

        # find images
        if len(args.img_filter)>0:
            imf = args.img_filter
        else:
            imf = None

        # Check with user if they REALLY mean to run without saving anything 
        if not (args.train or args.train_size):
            saving_something = args.save_png or args.save_tif or args.save_flows or args.save_ncolor or args.save_txt
                    
        device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)

        if args.pretrained_model is None or args.pretrained_model == 'None' or args.pretrained_model == 'False' or args.pretrained_model == '0':
            pretrained_model = False
        else:
            pretrained_model = args.pretrained_model
        
        model_type = None
        if pretrained_model and not os.path.exists(pretrained_model):
            model_type = pretrained_model if pretrained_model is not None else 'cyto'
            model_strings = models.get_user_models()
            all_models = models.MODEL_NAMES.copy() 
            all_models.extend(model_strings)
            if ~np.any([model_type == s for s in all_models]):
                model_type = 'cyto'
                logger.warning('pretrained model has incorrect path')
            if model_type=='nuclei':
                szmean = 17. 
            else:
                szmean = 30.
        builtin_size = model_type == 'cyto' or model_type == 'cyto2' or model_type == 'nuclei' or model_type=="cyto3"
        
        if len(args.image_path) > 0 and (args.train or args.train_size):
            raise ValueError('ERROR: cannot train model with single image input')

        if not args.train and not args.train_size:
            tic = time.time()
            if len(args.dir) > 0:
                image_names = io.get_image_files(args.dir, 
                                                args.mask_filter, 
                                                imf=imf,
                                                look_one_level_down=args.look_one_level_down)
            else:
                if os.path.exists(args.image_path):
                    image_names = [args.image_path]
                else:
                    raise ValueError(f'ERROR: no file found at {args.image_path}')
            nimg = len(image_names)
                
            cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
            cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
            logger.info('>>>> running cellpose on %d images using chan_to_seg %s and chan (opt) %s'%
                            (nimg, cstr0[channels[0]], cstr1[channels[1]]))
             
            # handle built-in model exceptions; bacterial ones get no size model 
            if builtin_size:
                model = models.Cellpose(gpu=gpu, device=device, model_type=model_type)                
            else:
                if args.all_channels:
                    channels = None  
                pretrained_model = None if model_type is not None else pretrained_model
                model = models.CellposeModel(gpu=gpu, device=device, 
                                             pretrained_model=pretrained_model,
                                             model_type=model_type)
            
            # handle diameters
            if args.diameter==0:
                if builtin_size:
                    diameter = None
                    logger.info('>>>> estimating diameter for each image')
                else:
                    logger.info('>>>> not using cyto, cyto2, or nuclei model, cannot auto-estimate diameter')
                    diameter = model.diam_labels
                    logger.info('>>>> using diameter %0.3f for all images'%diameter)
            else:
                diameter = args.diameter
                logger.info('>>>> using diameter %0.3f for all images'%diameter)
            
            
            tqdm_out = utils.TqdmToLogger(logger,level=logging.INFO)
            
            for image_name in tqdm(image_names, file=tqdm_out):
                image = io.imread(image_name)
                out = model.eval(image, channels=channels, diameter=diameter,
                                do_3D=args.do_3D,
                                augment=args.augment,
                                resample=(not args.no_resample),
                                flow_threshold=args.flow_threshold,
                                cellprob_threshold=args.cellprob_threshold,
                                stitch_threshold=args.stitch_threshold,
                                min_size=args.min_size,
                                invert=args.invert,
                                batch_size=args.batch_size,
                                interp=(not args.no_interp),
                                normalize=(not args.no_norm),
                                channel_axis=args.channel_axis,
                                z_axis=args.z_axis,
                                anisotropy=args.anisotropy)
                masks, flows = out[:2]
                if len(out) > 3:
                    diams = out[-1]
                else:
                    diams = diameter
                if args.exclude_on_edges:
                    masks = utils.remove_edge_masks(masks)
                if not args.no_npy:
                    io.masks_flows_to_seg(image, masks, flows, image_name, channels=channels, diams=diams)
                if saving_something:
                    io.save_masks(image, masks, flows, image_name, png=args.save_png, tif=args.save_tif,
                                  save_flows=args.save_flows,save_outlines=args.save_outlines,
                                  save_ncolor=args.save_ncolor,dir_above=args.dir_above,savedir=args.savedir,
                                  save_txt=args.save_txt,in_folders=args.in_folders, save_mpl=args.save_mpl)
                if args.save_rois:
                    io.save_rois(masks, image_name)
            logger.info('>>>> completed in %0.3f sec'%(time.time()-tic))
        else:
            
            test_dir = None if len(args.test_dir)==0 else args.test_dir
            output = io.load_train_test_data(args.dir, test_dir, imf, args.mask_filter, args.look_one_level_down)
            images, labels, image_names, test_images, test_labels, image_names_test = output

            # training with all channels
            if args.all_channels:
                img = images[0]
                if img.ndim==3:
                    nchan = min(img.shape)
                elif img.ndim==2:
                    nchan = 1
                channels = None 
            else:
                nchan = 2 
            
            # model path
            szmean = args.diam_mean
            if not os.path.exists(pretrained_model) and model_type is None:
                if not args.train:
                    error_message = 'ERROR: model path missing or incorrect - cannot train size model'
                    logger.critical(error_message)
                    raise ValueError(error_message)
                pretrained_model = False
                logger.info('>>>> training from scratch')
            if args.train:
                logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diam_mean)
                
            # initialize model
            model = models.CellposeModel(device=device,
                                         pretrained_model=pretrained_model if model_type is None else None,
                                         model_type=model_type, 
                                         diam_mean=szmean,
                                         nchan=nchan)
            
            # train segmentation model
            if args.train:
                cpmodel_path = train.train_seg(model.net, images, labels, train_files=image_names,
                                           test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                           learning_rate=args.learning_rate, 
                                           weight_decay=args.weight_decay,
                                           channels=channels,
                                           save_path=os.path.realpath(args.dir), 
                                           save_every=args.save_every,
                                           SGD=args.SGD,
                                           n_epochs=args.n_epochs,
                                           batch_size=args.batch_size, 
                                           min_train_masks=args.min_train_masks,
                                           model_name=args.model_name_out)
                model.pretrained_model = cpmodel_path
                logger.info('>>>> model trained and saved to %s'%cpmodel_path)

            # train size model
            if args.train_size:
                sz_model = models.SizeModel(cp_model=model, device=device)
                masks = [lbl[0] for lbl in labels]
                test_masks = [lbl[0] for lbl in test_labels] if test_labels is not None else test_labels
                # data has already been normalized and reshaped
                sz_model.params = train.train_size(model.net, model.pretrained_model, images, masks, test_images, test_masks, 
                                channels=channels, 
                                    batch_size=args.batch_size)
                if test_images is not None:
                    predicted_diams, diams_style = sz_model.eval(test_images, 
                                                                    channels=channels)
                    ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
                    cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
                    logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
                    np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]), 
                            {'predicted_diams': predicted_diams, 'diams_style': diams_style})

if __name__ == '__main__':
    main()
    
