import sys, os, argparse, glob, pathlib, time
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io, core, version_str, transforms


def main():
    parser = argparse.ArgumentParser(description='cellpose parameters')

    input_img_args = parser.add_argument_group("input image arguments")
    input_img_args.add_argument('--dir', default=[], type=str,
                                help='folder containing data to run or train on.')
    input_img_args.add_argument(
        '--image_path', default=[], type=str, help=
        'if given and --dir not given, run on single image instead of folder (cannot train with this option)'
    )
    input_img_args.add_argument(
        '--look_one_level_down', action='store_true',
        help='run processing on all subdirectories of current folder')
    input_img_args.add_argument('--img_filter', default=[], type=str,
                                help='end string for images to run on')
    input_img_args.add_argument(
        '--channel_axis', default=None, type=int,
        help='axis of image which corresponds to image channels')
    input_img_args.add_argument('--z_axis', default=None, type=int,
                                help='axis of image which corresponds to Z dimension')
    input_img_args.add_argument(
        '--chan', default=0, type=int, help=
        'channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument(
        '--chan2', default=0, type=int, help=
        'nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s'
    )
    input_img_args.add_argument('--invert', action='store_true',
                                help='invert grayscale channel')
    input_img_args.add_argument(
        '--all_channels', action='store_true', help=
        'use all channels in image if using own model and images with special channels')
    input_img_args.add_argument("--anisotropy", required=False, default=1.0, type=float,
                                help="anisotropy of volume in 3D")
    

    # algorithm settings
    algorithm_args = parser.add_argument_group("algorithm arguments")
    algorithm_args.add_argument('--sharpen_radius', required=False, default=0.0,
                                type=float, help='high-pass filtering radius. Default: %(default)s')
    algorithm_args.add_argument('--tile_norm', required=False, default=0, type=int,
                                help='tile normalization block size. Default: %(default)s')
    algorithm_args.add_argument('--nimg_per_tif', required=False, default=10, type=int,
                                help='number of crops in XY to save per tiff. Default: %(default)s')
    algorithm_args.add_argument('--crop_size', required=False, default=512, type=int,
                                help='size of random crop to save. Default: %(default)s')

    args = parser.parse_args()

    # find images
    if len(args.img_filter) > 0:
        imf = args.img_filter
    else:
        imf = None

    image_names = io.get_image_files(args.dir, "_masks", imf=imf,
                                     look_one_level_down=args.look_one_level_down)
    np.random.seed(0)
    nimg_per_tif = args.nimg_per_tif
    crop_size = args.crop_size
    os.makedirs(os.path.join(args.dir, 'train/'), exist_ok=True)
    pm = [(0, 1, 2, 3), (2, 0, 1, 3), (1, 0, 2, 3)]
    npm = ["YX", "ZY", "ZX"]
    for name in image_names:
        name0 = os.path.splitext(os.path.split(name)[-1])[0]
        img0 = io.imread(name)
        img0 = transforms.convert_image(img0, channels=[args.chan, args.chan2], channel_axis=args.channel_axis, z_axis=args.z_axis)
        for p in range(3):
            img = img0.transpose(pm[p]).copy()
            print(npm[p], img[0].shape)
            Ly, Lx = img.shape[1:3]
            imgs = img[np.random.permutation(img.shape[0])[:args.nimg_per_tif]]
            if args.anisotropy > 1.0:
                imgs = transforms.resize_image(imgs, Ly=int(args.anisotropy * Ly), Lx=Lx)
            for k, img in enumerate(imgs):
                if args.tile_norm:
                    img = transforms.normalize99_tile(img, blocksize=args.tile_norm)
                if args.sharpen_radius:
                    img = transforms.smooth_sharpen_img(img,
                                                        sharpen_radius=args.sharpen_radius)
                ly = 0 if Ly - crop_size <= 0 else np.random.randint(0, Ly - crop_size)
                lx = 0 if Lx - crop_size <= 0 else np.random.randint(0, Lx - crop_size)
                io.imsave(os.path.join(args.dir, f'train/{name0}_{npm[p]}_{k}.tif'),
                        img[ly:ly + args.crop_size, lx:lx + args.crop_size].squeeze())


if __name__ == '__main__':
    main()
