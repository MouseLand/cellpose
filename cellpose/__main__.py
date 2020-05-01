import numpy as np
import mxnet as mx
import os, argparse, glob, pathlib
import skimage
from natsort import natsorted

from . import utils, models, io


try:
    from cellpose import gui 
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

def get_image_files(folder):
    image_names = []
    image_names.extend(glob.glob(folder + '/*%s.png'%imf))
    image_names.extend(glob.glob(folder + '/*%s.jpg'%imf))
    image_names.extend(glob.glob(folder + '/*%s.jpeg'%imf))
    image_names.extend(glob.glob(folder + '/*%s.tif'%imf))
    image_names.extend(glob.glob(folder + '/*%s.tiff'%imf))
    image_names = natsorted(image_names)
    return image_names
        
def get_label_files(image_names, imf, mask_filter):
    nimg = len(image_names)
    label_names = [os.path.splitext(image_names[n])[0] for n in range(nimg)]
    if len(imf) > 0:
        label_names = [label_names[n][:-len(imf)] for n in range(nimg)]
    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    return label_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cellpose parameters')
    parser.add_argument('--train', action='store_true', help='train network using images in dir (not yet implemented)')
    parser.add_argument('--dir', required=False, 
                        default=[], type=str, help='folder containing data to run or train on')
    parser.add_argument('--img_filter', required=False, 
                        default=[], type=str, help='end string for images to run on')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu if mxnet with cuda installed')
    parser.add_argument('--do_3D', action='store_true',
                        help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')
    # settings for running cellpose
    parser.add_argument('--pretrained_model', required=False, 
                        default='cyto', type=str, help='model to use')
    #parser.add_argument('--unet', required=False, 
    #                    default=0, type=int, help='run standard unet instead of cellpose flow output')
    parser.add_argument('--chan', required=False, 
                        default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--chan2', required=False, 
                        default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    parser.add_argument('--diameter', required=False, 
                        default=30., type=float, help='cell diameter, if 0 cellpose will estimate for each image')
    parser.add_argument('--flow_threshold', required=False, 
                        default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--cellprob_threshold', required=False, 
                        default=0.0, type=float, help='cell probability threshold, centered at 0.0')
    parser.add_argument('--save_png', action='store_true', help='save masks as png')

    # settings for training
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

    args = parser.parse_args()

    if len(args.dir)==0:
        if not GUI_ENABLED:
            print('ERROR: %s'%GUI_ERROR)
            if GUI_IMPORT:
                print('GUI FAILED: GUI dependencies may not be installed, to install, run')
                print('     pip install cellpose[gui]')
        else:
            gui.run()

    else:
        use_gpu = False
        channels = [args.chan, args.chan2]

        # find images
        if len(args.img_filter)>0:
            imf = args.img_filter
        else:
            imf = ''


        image_names = get_image_files(args.dir)
        imn = []
        for im in image_names:
            imfile = os.path.splitext(im)[0]
            if len(imfile) > len(args.mask_filter):
                if imfile[-len(args.mask_filter):] != args.mask_filter:
                    imn.append(im)
            else:
                imn.append(im)
        image_names = imn
        nimg = len(image_names)
        images = [skimage.io.imread(image_names[n]) for n in range(nimg)]

        if args.use_gpu:
            use_gpu = utils.use_gpu()
        if use_gpu:
            device = mx.gpu()
        else:
            device = mx.cpu()
        print('>>>> using %s'%(['CPU', 'GPU'][use_gpu]))
        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')              

        if not args.train:
            if not (args.pretrained_model=='cyto' or args.pretrained_model=='nuclei'):
                cpmodel_path = args.pretrained_model
                if not os.path.exists(cpmodel_path):
                    print('model path does not exist, using cyto model')
                    args.pretrained_model = 'cyto'

            if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei':
                model = models.Cellpose(device=device, model_type=args.pretrained_model)
                    
                if args.diameter==0:
                    diameter = None
                    print('>>>> estimating diameter for each image')
                else:
                    diameter = args.diameter
                    print('>>>> using diameter %0.2f for all images'%diameter)

                cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
                cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
                print('running cellpose on %d images using chan_to_seg %s and chan (opt) %s'%
                        (nimg, cstr0[channels[0]], cstr1[channels[1]]))
                
                masks, flows, _, diams = model.eval(images, channels=channels, diameter=diameter,
                                                    do_3D=args.do_3D,
                                                    flow_threshold=args.flow_threshold,
                                                    cellprob_threshold=args.cellprob_threshold)
                
            else:
                if args.all_channels:
                    channels = None  
                if args.diameter==0:
                    print('>>>> using user-specified model, no auto-diameter estimation available')
                    diameter = 30.
                else:
                    diameter = args.diameter
                model = models.CellposeModel(device=device, pretrained_model=cpmodel_path)
                masks, flows, _ = model.eval(images, channels=channels, diameter=diameter,
                                             do_3D=args.do_3D,
                                             flow_threshold=args.flow_threshold,
                                             cellprob_threshold=args.cellprob_threshold)
                diams = diameter * np.ones(len(images)) 
                  
            print('>>>> saving results')
            io.masks_flows_to_seg(images, masks, flows, diams, image_names, channels)
            if args.save_png:
                io.save_to_png(images, masks, flows, image_names)
                    
        else:
            if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei':
                cpmodel_path = os.fspath(model_dir.joinpath('%s_0'%(args.pretrained_model)))
                if args.pretrained_model=='cyto':
                    szmean = 27.
                else:
                    szmean = 15.
            else:
                cpmodel_path = os.fspath(args.pretrained_model)
                szmean = 27.
            
            if args.all_channels:
                channels = None  

            label_names = get_label_files(image_names, imf, args.mask_filter)
            nimg = len(image_names)
            labels = [skimage.io.imread(label_names[n]) for n in range(nimg)]
            if not os.path.exists(cpmodel_path):
                cpmodel_path = False
                print('>>>> training from scratch')
                if args.diameter==0:
                    rescale = False 
                    print('>>>> median diameter set to 0 => no rescaling during training')
                else:
                    rescale = True
                    szmean = args.diameter * (np.pi**0.5/2)
            else:
                rescale = True
                args.diameter = szmean / (np.pi**0.5/2)
                print('>>>> training starting with pretrained_model %s'%cpmodel_path)
            if rescale:
                print('>>>> rescaling diameter for each training image to %0.1f'%args.diameter)
                
            

            test_images, test_labels = None, None
            if len(args.test_dir) > 0:
                image_names_test = get_image_files(args.test_dir)
                label_names_test = get_label_files(image_names_test, imf, args.mask_filter)
                nimg = len(image_names_test)
                test_images = [skimage.io.imread(image_names_test[n]) for n in range(nimg)]
                test_labels = [skimage.io.imread(label_names_test[n]) for n in range(nimg)]
            #print('>>>> %s model'%(['cellpose', 'unet'][args.unet]))    
            model = models.CellposeModel(device=device,
                                         pretrained_model=cpmodel_path, 
                                         diam_mean=szmean)
            
            model.train(images, labels, test_images, test_labels, learning_rate=args.learning_rate,
                        channels=channels, save_path=os.path.realpath(args.dir), rescale=rescale)