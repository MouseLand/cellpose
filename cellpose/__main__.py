import sys, os, argparse, glob, pathlib, time
import subprocess

import numpy as np
from natsort import natsorted
from tqdm import tqdm

from cellpose import utils, models, io

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
logger = logging.getLogger(__name__)

def confirm_prompt(question):
    reply = None
    while reply not in ("", "y", "n"):
        reply = input(f"{question} (y/n): ").lower()
    return (reply in ("", "y"))

# settings re-grouped a bit
def main():
    parser = argparse.ArgumentParser(description='cellpose parameters')
    
    # settings for CPU vs GPU
    parser.add_argument('--use_gpu', action='store_true', help='use gpu if mxnet with cuda installed')
    parser.add_argument('--check_mkl', action='store_true', help='check if mkl working')
    parser.add_argument('--mkldnn', action='store_true', help='for mxnet, force MXNET_SUBGRAPH_BACKEND = "MKLDNN"')
        
    # settings for locating and formatting images
    parser.add_argument('--dir', required=False, 
                        default=[], type=str, help='folder containing data to run or train on')
    parser.add_argument('--look_one_level_down', action='store_true', help='')
    parser.add_argument('--mxnet', action='store_true', help='use mxnet')
    parser.add_argument('--img_filter', required=False, 
                        default=[], type=str, help='end string for images to run on')
    parser.add_argument('--channel_axis', required=False, 
                        default=None, type=int, help='axis of image which corresponds to image channels')
    parser.add_argument('--z_axis', required=False, 
                        default=None, type=int, help='axis of image which corresponds to Z dimension')
    parser.add_argument('--chan', required=False,
                        default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--chan2', required=False, 
                        default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--invert', required=False, action='store_true', help='invert grayscale channel')
    parser.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    
    # model settings 
#     parser.add_argument('--model_dir', required=False,
#                         default=None, type=str, help='directory with built-in models, default is $HOME/.cellpose/models/')
    parser.add_argument('--unet', required=False,
                        default=0, type=int, help='run standard unet instead of cellpose flow output')
    parser.add_argument('--nclasses', required=False,
                        default=3, type=int, 
                        help='if running unet, choose 2 or 3; if training skel, choose 4; standard Cellpose uses 3')

    # cellpose algorithm settings
    parser.add_argument('--skel', action='store_true', help='flag to enable "skeletonized" algorithm (disabled by default)')
    parser.add_argument('--fast_mode', action='store_true', help="make code run faster by turning off 4 network averaging")
    parser.add_argument('--resample', action='store_true', help="run dynamics on full image (slower for images with large diameters)")
    parser.add_argument('--no_interp', action='store_true', help='do not interpolate when running dynamics (was default)')
    parser.add_argument('--do_3D', action='store_true', help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')
    parser.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use')
    parser.add_argument('--diameter', required=False, default=30., type=float, 
                        help='cell diameter, if 0 cellpose will estimate for each image')
    parser.add_argument('--stitch_threshold', required=False, default=0.0, type=float,
                        help='compute masks in 2D then stitch together masks with IoU>0.9 across planes')
    parser.add_argument('--flow_threshold', required=False, 
                        default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--dist_threshold', required=False, 
                        default=0, type=float, help='cell distance threshold')
    parser.add_argument('--diam_threshold', required=False, default=12.0, type=float, 
                        help='cell diameter threshold for upscaling before mask rescontruction, default 12.')
    parser.add_argument('--exclude_on_edges', action='store_true', help='discard masks which touch edges of image')
    
    # output settings
    parser.add_argument('--save_png', action='store_true', help='save masks as png and outlines as text file for ImageJ')
    parser.add_argument('--save_tif', action='store_true', help='save masks as tif and outlines as text file for ImageJ')
    parser.add_argument('--no_npy', action='store_false', help='suppress saving of npy')
    parser.add_argument('--savedir', required=False, 
                        default=None, type=str, help='folder to which segmentation results will be saved (defaults to input image directory)')
    parser.add_argument('--dir_above', action='store_true', help='save output folders adjacent to image folder instead of inside it (off by default)')
    parser.add_argument('--in_folders', action='store_true', help='flag to save output in folders (off by default)')
    parser.add_argument('--save_flows', action='store_true', help='whether or not to save RGB images of flows when masks are saved (disabled by default)')
    parser.add_argument('--save_outlines', action='store_true', help='whether or not to save RGB outline images when masks are saved (disabled by default)')
    parser.add_argument('--save_ncolor', action='store_true', help='whether or not to save minimal "n-color" masks (disabled by default')
    parser.add_argument('--save_txt', action='store_true', help='flag to enable txt outlines for ImageJ (disabled by default)')

    # training settings
    parser.add_argument('--train', action='store_true', help='train network using images in dir')
    parser.add_argument('--train_size', action='store_true', help='train size network at end of training')
    parser.add_argument('--mask_filter', required=False, 
                        default='_masks', type=str, help='end string for masks to run on')
    parser.add_argument('--test_dir', required=False, 
                        default=[], type=str, help='folder containing test data (optional)')
    parser.add_argument('--learning_rate', required=False, 
                        default=0.2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False, 
                        default=500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', required=False, 
                        default=8, type=int, help='batch size')
    parser.add_argument('--residual_on', required=False, 
                        default=1, type=int, help='use residual connections')
    parser.add_argument('--style_on', required=False, 
                        default=1, type=int, help='use style vector')
    parser.add_argument('--concatenation', required=False, 
                        default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    parser.add_argument('--save_every', required=False,
                        default=100, type=int, help='number of epochs to skip between saves')
    parser.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')
    
    # misc settings
    parser.add_argument('--verbose', action='store_true', help='flag to output extra information (e.g. diameter metrics) for debugging and fine-tuning parameters')
    parser.add_argument('--testing', action='store_true', help='flag to suppress CLI user confirmation for saving output; for test scripts')



    args = parser.parse_args()
    
    # skel changes not implemented for mxnet. Full parity for cpu/gpu in pytorch. 
    if args.skel and args.mxnet:
        logger.info('>>>> Skel only implemented in pytorch.')
        confirm = confirm_prompt('Continue with skel set to false?')
        if not confirm:
            exit()
        else:
            logger.info('>>>> Skel set to false.')
            args.skel = False

    # For now, skel version is not compatible with 3D. WIP. 
    if args.skel and args.do_3D:
        logger.info('>>>> Skel not yet compatible with 3D segmentation.')
        confirm = confirm_prompt('Continue with skel set to false?')
        if not confirm:
            exit()
        else:
            logger.info('>>>> Skel set to false.')
            args.skel = False
    
    # skel model needs 4 classes. Would prefer a more elegant way to automaticaly update the flow fields
    # instead of users deleting them manually - a check on the number of channels, maybe, or just use
    # the yes/no prompt to ask the user if they want their flow fields in the given directory to be deleted. 
    # would also need the look_one_level_down optionally toggled...
    if args.skel and args.train:
        logger.info('>>>> Training skel model. Setting nclasses to 4.')
        logger.info('>>>> Make sure your flow fields are deleted and re-computed.')
        args.nclasses = 4
    
                
    if args.check_mkl:
        mkl_enabled = models.check_mkl((not args.mxnet))
    else:
        mkl_enabled = True

    if not args.train and (mkl_enabled and args.mkldnn):
        os.environ["MXNET_SUBGRAPH_BACKEND"]="MKLDNN"
    else:
        os.environ["MXNET_SUBGRAPH_BACKEND"]=""
    
    if len(args.dir)==0:
        if not GUI_ENABLED:
            logger.critical('ERROR: %s'%GUI_ERROR)
            if GUI_IMPORT:
                logger.critical('GUI FAILED: GUI dependencies may not be installed, to install, run')
                logger.critical('     pip install cellpose[gui]')
        else:
            gui.run()

    else:
        use_gpu = False
        channels = [args.chan, args.chan2]

        # find images
        if len(args.img_filter)>0:
            imf = args.img_filter
        else:
            imf = None


        # Check with user
        if not (args.train or args.train_size):
            saving_something = args.save_png or args.save_tif or args.save_flows or args.save_ncolor or args.save_txt
            if not (saving_something or args.testing): 
                logger.info('>>>> Running without saving any output.')
                confirm = confirm_prompt('Proceed Anyway?')
                if not confirm:
                    exit()
                    
                    
        device, gpu = models.assign_device((not args.mxnet), args.use_gpu)

        if not args.train and not args.train_size:
            tic = time.time()
            if not (args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2'):
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
                    
            if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2':
                if args.mxnet and args.pretrained_model=='cyto2':
                    logger.warning('cyto2 model not available in mxnet, using cyto model')
                    args.pretrained_model = 'cyto'
                model = models.Cellpose(gpu=gpu, device=device, model_type=args.pretrained_model, 
                                            torch=(not args.mxnet))
            else:
                if args.all_channels:
                    channels = None  
                model = models.CellposeModel(gpu=gpu, device=device, 
                                             pretrained_model=cpmodel_path,
                                             torch=(not args.mxnet))

            if args.diameter==0:
                if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2':
                    diameter = None
                    logger.info('>>>> estimating diameter for each image')
                else:
                    logger.info('>>>> using user-specified model, no auto-diameter estimation available')
                    diameter = model.diam_mean
            else:
                diameter = args.diameter
                logger.info('>>>> using diameter %0.2f for all images'%diameter)
            
            
            tqdm_out = utils.TqdmToLogger(logger,level=logging.INFO)
            for image_name in tqdm(image_names, file=tqdm_out):
                image = io.imread(image_name)
                out = model.eval(image, channels=channels, diameter=diameter,
                                do_3D=args.do_3D, net_avg=(not args.fast_mode),
                                augment=False,
                                resample=args.resample,
                                flow_threshold=args.flow_threshold,
                                dist_threshold=args.dist_threshold,
                                diam_threshold=args.diam_threshold,
                                invert=args.invert,
                                batch_size=args.batch_size,
                                interp=(not args.no_interp),
                                channel_axis=args.channel_axis,
                                z_axis=args.z_axis,
                                skel=args.skel,
                                verbose=args.verbose)
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
            if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2':
                if args.mxnet and args.pretrained_model=='cyto2':
                    logger.warning('cyto2 model not available in mxnet, using cyto model')
                    args.pretrained_model = 'cyto'
                cpmodel_path = models.model_path(args.pretrained_model, 0, not args.mxnet)
                if args.pretrained_model=='cyto':
                    szmean = 30.
                else:
                    szmean = 17.
            else:
                cpmodel_path = os.fspath(args.pretrained_model)
                szmean = 30.
            
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
                if args.diameter==0:
                    rescale = False 
                    logger.info('>>>> median diameter set to 0 => no rescaling during training')
                else:
                    rescale = True
                    szmean = args.diameter 
            else:
                rescale = True
                args.diameter = szmean 
                logger.info('>>>> pretrained model %s is being used'%cpmodel_path)
                args.residual_on = 1
                args.style_on = 1
                args.concatenation = 0
            if rescale and args.train:
                logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diameter)
                
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
                                            torch=(not args.mxnet),
                                            pretrained_model=cpmodel_path, 
                                            diam_mean=szmean,
                                            residual_on=args.residual_on,
                                            style_on=args.style_on,
                                            concatenation=args.concatenation,
                                            nchan=nchan,
                                            skel=args.skel)
            
            # train segmentation model
            if args.train:
                cpmodel_path = model.train(images, labels, train_files=image_names,
                                           test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                           learning_rate=args.learning_rate, channels=channels,
                                           save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                           save_each=args.save_each,
                                           rescale=rescale,n_epochs=args.n_epochs,
                                           batch_size=args.batch_size, skel=args.skel)
                model.pretrained_model = cpmodel_path
                logger.info('>>>> model trained and saved to %s'%cpmodel_path)

            # train size model
            if args.train_size:
                sz_model = models.SizeModel(cp_model=model, device=device)
                sz_model.train(images, labels, test_images, test_labels, channels=channels, batch_size=args.batch_size)
                if test_images is not None:
                    predicted_diams, diams_style = sz_model.eval(test_images, channels=channels)
                    if test_labels[0].ndim>2:
                        tlabels = [lbl[0] for lbl in test_labels]
                    else:
                        tlabels = test_labels 
                    ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
                    cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
                    logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
                    np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]), 
                            {'predicted_diams': predicted_diams, 'diams_style': diams_style})

if __name__ == '__main__':
    main()
    
