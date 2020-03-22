import numpy as np
import mxnet as mx
import os, argparse, glob
import skimage
from natsort import natsorted

from cellpose import utils, models, gui

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

    # settings for running cellpose
    parser.add_argument('--pretrained_model', required=False, 
                        default='cyto', type=str, help='model to use')
    parser.add_argument('--unet', required=False, 
                        default=0, type=int, help='run standard unet instead of cellpose flow output')
    parser.add_argument('--chan', required=False, 
                        default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--chan2', required=False, 
                        default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    parser.add_argument('--diameter', required=False, 
                        default=30., type=float, help='cell diameter, if 0 cellpose will estimate for each image')
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
        try:
            from cellpose import gui
        except:
            print('pyqt is not installed')
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
        #nimg = 
        images = [skimage.io.imread(image_names[n]) for n in range(nimg)]

        if args.use_gpu:
            use_gpu = utils.use_gpu()
        if use_gpu:
            device = mx.gpu()
        else:
            device = mx.cpu()
        print('>>>> using %s'%(['CPU', 'GPU'][use_gpu]))

        if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei':
            cp_path = os.path.dirname(os.path.realpath(__file__))
            if not args.train:
                model_list = ['%s_%d'%(args.pretrained_model, i) for i in range(4)]
                cpmodel_path = [os.path.abspath(os.path.join(cp_path, 'models/', model_list[i]))
                                    for i in range(len(model_list))]
            else:
                model_list = ['%s_0'%(args.pretrained_model)]
                cpmodel_path = os.path.join(cp_path, 'models/', model_list[0])
            
            szmodel_path = os.path.abspath(os.path.join(cp_path, 'models/', 'size_%s_0.npy'%args.pretrained_model))
            if args.pretrained_model=='cyto':
                szmean = 27.
            else:
                szmean = 15.
        else:
            cpmodel_path = args.pretrained_model
            szmodel_path = None
            szmean = 27.
            if args.all_channels:
                channels = None                

        if not args.train:
            model = models.Cellpose(device=device, 
                                    pretrained_model=cpmodel_path,
                                    pretrained_size=szmodel_path,
                                    diam_mean=szmean
                                   )
            if args.diameter==0:
                diameter = None
                if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei':
                    print('>>>> estimating diameter for each image')
                else:
                    print('>>>> using user-specified model, no auto-diameter estimation available')
            else:
                diameter = args.diameter
                print('using diameter %0.2f for all images'%diameter)
            cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
            cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
            print('running cellpose on %d images using chan_to_seg %s and chan (opt) %s'%
                    (nimg, cstr0[channels[0]], cstr1[channels[1]]))
            
            masks, flows, _, diams = model.eval(images, channels=channels, diameter=diameter)
            print('>>>> saving results')
            utils.masks_flows_to_seg(images, masks, flows, diams, channels, image_names)
            if args.save_png:
                utils.save_to_png(images, masks, flows, image_names)
                    
        else:
            label_names = get_label_files(image_names, imf, args.mask_filter)
            nimg = len(image_names)
            labels = [skimage.io.imread(label_names[n]) for n in range(nimg)]
            if not os.path.exists(cpmodel_path):
                cpmodel_path = None
                print('>>>> training from scratch')
            else:
                print('>>>> training starting with pretrained_model %s'%cpmodel_path)

            test_images, test_labels = None, None
            if len(args.test_dir) > 0:
                image_names_test = get_image_files(args.test_dir)
                label_names_test = get_label_files(image_names_test, imf, args.mask_filter)
                nimg = len(image_names_test)
                test_images = [skimage.io.imread(image_names_test[n]) for n in range(nimg)]
                test_labels = [skimage.io.imread(label_names_test[n]) for n in range(nimg)]
            print('>>>> %s model'%(['cellpose', 'unet'][args.unet]))    
            model = models.CellposeModel(device=device, unet=args.unet, pretrained_model=cpmodel_path)
            model.train(images, labels, test_images, test_labels, learning_rate=args.learning_rate,
                        channels=channels, save_path=os.path.realpath(args.dir))