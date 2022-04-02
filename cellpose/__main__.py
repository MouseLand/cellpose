import sys, os, argparse, glob, pathlib, time
import subprocess

import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io, core

try:
    from cellpose.gui import gui 
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
    parser = argparse.ArgumentParser(description='cellpose parameters')
    
    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("hardware arguments")
    hardware_args.add_argument('--use_gpu', action='store_true', help='use gpu if torch with cuda installed')
    hardware_args.add_argument('--check_mkl', action='store_true', help='check if mkl working')
        
    # settings for locating and formatting images
    input_img_args = parser.add_argument_group("input image arguments")
    input_img_args.add_argument('--dir',
                        default=[], type=str, help='folder containing data to run or train on.')
    input_img_args.add_argument('--look_one_level_down', action='store_true', help='run processing on all subdirectories of current folder')
    input_img_args.add_argument('--img_filter',
                        default=[], type=str, help='end string for images to run on')
    input_img_args.add_argument('--channel_axis',
                        default=None, type=int, help='axis of image which corresponds to image channels')
    input_img_args.add_argument('--z_axis',
                        default=None, type=int, help='axis of image which corresponds to Z dimension')
    input_img_args.add_argument('--chan',
                        default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument('--chan2',
                        default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument('--invert', action='store_true', help='invert grayscale channel')
    input_img_args.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    
    # model settings 
    model_args = parser.add_argument_group("model arguments")
    model_args.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use for running or starting training')
    model_args.add_argument('--unet', action='store_true', help='run standard unet instead of cellpose flow output')
    model_args.add_argument('--nclasses',default=3, type=int, help='if running unet, choose 2 or 3; cellpose always uses 3')

    # algorithm settings
    algorithm_args = parser.add_argument_group("algorithm arguments")
    algorithm_args.add_argument('--no_resample', action='store_true', help="disable dynamics on full image (makes algorithm faster for images with large diameters)")
    algorithm_args.add_argument('--net_avg', action='store_true', help='run 4 networks instead of 1 and average results')
    algorithm_args.add_argument('--no_interp', action='store_true', help='do not interpolate when running dynamics (was default)')
    algorithm_args.add_argument('--do_3D', action='store_true', help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')
    algorithm_args.add_argument('--diameter', required=False, default=30., type=float, 
                        help='cell diameter, if 0 will use the diameter of the training labels used in the model, or with built-in model will estimate diameter for each image')
    algorithm_args.add_argument('--stitch_threshold', required=False, default=0.0, type=float, help='compute masks in 2D then stitch together masks with IoU>0.9 across planes')
    algorithm_args.add_argument('--fast_mode', action='store_true', help='now equivalent to --no_resample; make code run faster by turning off resampling')
    
    algorithm_args.add_argument('--flow_threshold', default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step. Default: %(default)s')
    algorithm_args.add_argument('--cellprob_threshold', default=0, type=float, help='cellprob threshold, default is 0, decrease to find more and larger masks')
    
    algorithm_args.add_argument('--anisotropy', required=False, default=1.0, type=float,
                        help='anisotropy of volume in 3D')
    algorithm_args.add_argument('--exclude_on_edges', action='store_true', help='discard masks which touch edges of image')
    
    # output settings
    output_args = parser.add_argument_group("output arguments")
    output_args.add_argument('--save_png', action='store_true', help='save masks as png and outlines as text file for ImageJ')
    output_args.add_argument('--save_tif', action='store_true', help='save masks as tif and outlines as text file for ImageJ')
    output_args.add_argument('--no_npy', action='store_true', help='suppress saving of npy')
    output_args.add_argument('--savedir',
                        default=None, type=str, help='folder to which segmentation results will be saved (defaults to input image directory)')
    output_args.add_argument('--dir_above', action='store_true', help='save output folders adjacent to image folder instead of inside it (off by default)')
    output_args.add_argument('--in_folders', action='store_true', help='flag to save output in folders (off by default)')
    output_args.add_argument('--save_flows', action='store_true', help='whether or not to save RGB images of flows when masks are saved (disabled by default)')
    output_args.add_argument('--save_outlines', action='store_true', help='whether or not to save RGB outline images when masks are saved (disabled by default)')
    output_args.add_argument('--save_ncolor', action='store_true', help='whether or not to save minimal "n-color" masks (disabled by default')
    output_args.add_argument('--save_txt', action='store_true', help='flag to enable txt outlines for ImageJ (disabled by default)')

    # training settings
    training_args = parser.add_argument_group("training arguments")
    training_args.add_argument('--train', action='store_true', help='train network using images in dir')
    training_args.add_argument('--train_size', action='store_true', help='train size network at end of training')
    training_args.add_argument('--test_dir',
                        default=[], type=str, help='folder containing test data (optional)')
    training_args.add_argument('--mask_filter',
                        default='_masks', type=str, help='end string for masks to run on. Default: %(default)s')
    training_args.add_argument('--diam_mean',
                        default=30., type=float, help='mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0')
    training_args.add_argument('--learning_rate',
                        default=0.2, type=float, help='learning rate. Default: %(default)s')
    training_args.add_argument('--weight_decay',
                        default=0.00001, type=float, help='weight decay. Default: %(default)s')
    training_args.add_argument('--n_epochs',
                        default=500, type=int, help='number of epochs. Default: %(default)s')
    training_args.add_argument('--batch_size',
                        default=8, type=int, help='batch size. Default: %(default)s')
    training_args.add_argument('--min_train_masks',
                        default=5, type=int, help='minimum number of masks a training image must have to be used. Default: %(default)s')
    training_args.add_argument('--residual_on',
                        default=1, type=int, help='use residual connections')
    training_args.add_argument('--style_on',
                        default=1, type=int, help='use style vector')
    training_args.add_argument('--concatenation',
                        default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    training_args.add_argument('--save_every',
                        default=100, type=int, help='number of epochs to skip between saves. Default: %(default)s')
    training_args.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')
    
    # misc settings
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')
    
    args = parser.parse_args()

    if args.check_mkl:
        mkl_enabled = models.check_mkl()
    else:
        mkl_enabled = True
    
    if len(args.dir)==0:
        if not GUI_ENABLED:
            print('GUI ERROR: %s'%GUI_ERROR)
            if GUI_IMPORT:
                print('GUI FAILED: GUI dependencies may not be installed, to install, run')
                print('     pip install cellpose[gui]')
        else:
            gui.run()

    else:
        if args.verbose:
            from .io import logger_setup
            logger, log_file = logger_setup()
        else:
            print('>>>> !NEW LOGGING SETUP! To see cellpose progress, set --verbose')
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
                    
        device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu)

        #define available model names, right now we have three broad categories 
        builtin_model = np.any([args.pretrained_model==s for s in models.MODEL_NAMES])
        cytoplasmic = 'cyto' in args.pretrained_model
        nuclear = 'nuclei' in args.pretrained_model
        
        if not args.train and not args.train_size:
            tic = time.time()
            if not builtin_model:
                cpmodel_path = args.pretrained_model
                if not os.path.exists(cpmodel_path):
                    logger.warning('model path does not exist, using cyto model')
                    args.pretrained_model = 'cyto'

            image_names = io.get_image_files(args.dir, 
                                             args.mask_filter, 
                                             imf=imf,
                                             look_one_level_down=args.look_one_level_down)
            nimg = len(image_names)
                
            cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
            cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
            logger.info('>>>> running cellpose on %d images using chan_to_seg %s and chan (opt) %s'%
                            (nimg, cstr0[channels[0]], cstr1[channels[1]]))
             
            # handle built-in model exceptions; bacterial ones get no size model 
            if builtin_model:
                model = models.Cellpose(gpu=gpu, device=device, model_type=args.pretrained_model, 
                                                net_avg=(not args.fast_mode or args.net_avg))
                
            else:
                if args.all_channels:
                    channels = None  
                model = models.CellposeModel(gpu=gpu, device=device, 
                                             pretrained_model=cpmodel_path,
                                             net_avg=False)
            
            # handle diameters
            if args.diameter==0:
                if builtin_model:
                    diameter = None
                    logger.info('>>>> estimating diameter for each image')
                else:
                    logger.info('>>>> not using cyto or nuclei model, cannot auto-estimate diameter')
                    diameter = model.diam_labels
                    logger.info('>>>> using diameter %0.3f for all images'%diameter)
            else:
                diameter = args.diameter
                logger.info('>>>> using diameter %0.3f for all images'%diameter)
            
            
            tqdm_out = utils.TqdmToLogger(logger,level=logging.INFO)
            
            for image_name in tqdm(image_names, file=tqdm_out):
                image = io.imread(image_name)
                out = model.eval(image, channels=channels, diameter=diameter,
                                do_3D=args.do_3D, net_avg=(not args.fast_mode or args.net_avg),
                                augment=False,
                                resample=(not args.no_resample and not args.fast_mode),
                                flow_threshold=args.flow_threshold,
                                cellprob_threshold=args.cellprob_threshold,
                                stitch_threshold=args.stitch_threshold,
                                invert=args.invert,
                                batch_size=args.batch_size,
                                interp=(not args.no_interp),
                                channel_axis=args.channel_axis,
                                z_axis=args.z_axis,
                                anisotropy=args.anisotropy,
                                model_loaded=True)
                masks, flows = out[:2]
                if len(out) > 3:
                    diams = out[-1]
                else:
                    diams = diameter
                if args.exclude_on_edges:
                    masks = utils.remove_edge_masks(masks)
                if not args.no_npy:
                    io.masks_flows_to_seg(image, masks, flows, diams, image_name, channels)
                if saving_something:
                    io.save_masks(image, masks, flows, image_name, png=args.save_png, tif=args.save_tif,
                                  save_flows=args.save_flows,save_outlines=args.save_outlines,
                                  save_ncolor=args.save_ncolor,dir_above=args.dir_above,savedir=args.savedir,
                                  save_txt=args.save_txt,in_folders=args.in_folders)
            logger.info('>>>> completed in %0.3f sec'%(time.time()-tic))
        else:
            if builtin_model:
                cpmodel_path = models.model_path(args.pretrained_model, 0)
                if cytoplasmic:
                    szmean = 30.
                elif nuclear:
                    szmean = 17.
            else:
                cpmodel_path = os.fspath(args.pretrained_model)
            
            test_dir = None if len(args.test_dir)==0 else args.test_dir
            output = io.load_train_test_data(args.dir, test_dir, imf, args.mask_filter, args.unet, args.look_one_level_down)
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
            if not os.path.exists(cpmodel_path):
                if not args.train:
                    error_message = 'ERROR: model path missing or incorrect - cannot train size model'
                    logger.critical(error_message)
                    raise ValueError(error_message)
                cpmodel_path = False
                logger.info('>>>> training from scratch')
                
                szmean = args.diam_mean
            else:
                args.diam_mean = szmean 
                logger.info('>>>> pretrained model %s is being used'%cpmodel_path)
                args.residual_on = 1
                args.style_on = 1
                args.concatenation = 0
            if args.train:
                logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diam_mean)
                
            # initialize model
            if args.unet:
                model = core.UnetModel(device=device,
                                        pretrained_model=cpmodel_path, 
                                        diam_mean=szmean,
                                        residual_on=args.residual_on,
                                        style_on=args.style_on,
                                        concatenation=args.concatenation,
                                        nclasses=args.nclasses,
                                        nchan=nchan)
            else:
                model = models.CellposeModel(device=device,
                                            pretrained_model=cpmodel_path, 
                                            diam_mean=szmean,
                                            residual_on=args.residual_on,
                                            style_on=args.style_on,
                                            concatenation=args.concatenation,
                                            nchan=nchan)
            
            # train segmentation model
            if args.train:
                cpmodel_path = model.train(images, labels, train_files=image_names,
                                           test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                           learning_rate=args.learning_rate, 
                                           weight_decay=args.weight_decay,
                                           channels=channels,
                                           save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                           save_each=args.save_each,
                                           n_epochs=args.n_epochs,
                                           batch_size=args.batch_size, 
                                           min_train_masks=args.min_train_masks)
                model.pretrained_model = cpmodel_path
                logger.info('>>>> model trained and saved to %s'%cpmodel_path)

            # train size model
            if args.train_size:
                sz_model = models.SizeModel(cp_model=model, device=device)
                masks = [lbl[0] for lbl in labels]
                test_masks = [lbl[0] for lbl in test_labels] if test_labels is not None else test_labels
                # data has already been normalized and reshaped
                sz_model.train(images, masks, test_images, test_masks, 
                                channels=None, normalize=False,
                                    batch_size=args.batch_size)
                if test_images is not None:
                    predicted_diams, diams_style = sz_model.eval(test_images, 
                                                                    channels=None,
                                                                    normalize=False)
                    ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
                    cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
                    logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
                    np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]), 
                            {'predicted_diams': predicted_diams, 'diams_style': diams_style})

if __name__ == '__main__':
    main()
    
