"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Heavily modified by Carsen Stringer for general datasets (12/2019)
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python run_maskrcnn.py --dataset=/path/to/dataset --weights=imagenet

    # dataset should have a train and test folder

"""
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os, sys, datetime, glob,pdb
import numpy as np
#np.random.bit_generator = np.random._bit_generator
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("/groups/pachitariu/pachitariulab/code/github/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import matplotlib.pyplot as plt

from stardist import matching

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#dm11_root = '/home/carsen/dm11/'
dm11_root = '/groups/pachitariu/pachitariulab/'
basedir = '/groups/pachitariu/home/stringerc/models' # where to save outputs
DEFAULT_LOGS_DIR = os.path.join(basedir, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(basedir, "maskrcnn/")

############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus
    
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1500

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 300

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400
    
    


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################
import glob
class NucleusDataset(utils.Dataset):
    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        fs = glob.glob(os.path.join(dataset_dir, '*_img.tif'))
        image_ids = np.arange(0, len(fs), 1, int)
        #assert subset in ["train", "val"]
        val = np.zeros(len(image_ids), np.bool)
        val[np.arange(0,len(image_ids),4,int)] = True
        if subset == "val":
            image_ids = image_ids[::8]#np.arange(0,81,1,int)
        else:
            # Get image ids from directory names
            if subset == "train":
                image_ids = image_ids[~val]
        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, "%03d_img.tif"%image_id))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        m = skimage.io.imread(info['path'][:-7]+'masks.tif')
        mask = []
        for k in range(m.max()):
            mask.append(m==(k+1))
        mask = np.stack(mask,axis=-1).astype(np.bool)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.5, 1.5)),
        #iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    
    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.HEAD_EPOCHS,
                augmentation=augmentation,
                layers='heads',
                )

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.TRAIN_EPOCHS,
                augmentation=augmentation,
                layers='all',
                )


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, RESULTS_DIR=RESULTS_DIR):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))
    #config.BATCH_SIZE = 1
    #config.IMAGES_PER_GPU = 1
    #config.GPU_COUNT = 1
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, '')
    dataset.prepare()
    # Load over images
    submission = []
    masks = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        #source_id = dataset.image_info[image_id]["id"]
        #rle = mask_to_rle(source_id, r["masks"], r["scores"])
        masks.append(r["masks"])
        #submission.append(rle)
        # Save image with masks
        #visualize.display_instances(
        #    image, r['rois'], r['masks'], r['class_ids'],
        #    dataset.class_names, r['scores'],
        #    show_bbox=False, show_mask=False,
        #    title="Predictions")
        #plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to npy file
    file_path = os.path.join(submit_dir, "overlapping_masks.npy")
    np.save(file_path, {'masks': masks})

    print("Saved to ", submit_dir)

    return masks

def remove_overlaps(masks, cellpix, medians):
    """ replace overlapping mask pixels with mask id of closest mask
        masks = Nmasks x Ly x Lx
    """
    overlaps = np.array(np.nonzero(cellpix>1.5)).T
    dists = ((overlaps[:,:,np.newaxis] - medians.T)**2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:,0], overlaps[:,1]] = 0
    masks[tocell, overlaps[:,0], overlaps[:,1]] = 1

    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1,masks.shape[0]+1,1,int)[:,np.newaxis,np.newaxis]
    masks = masks.sum(axis=0)
    return masks

def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--LR', default=0.001, type=float, required=False,
                        metavar="learning rate",
                        help="initial learning rate")
    parser.add_argument('--nepochs', default = 200, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default = 2, type=int, help='batch_size')

    args = parser.parse_args()

    nepochs = args.nepochs
    batch_size = args.batch_size
    learning_rate = args.LR

    # Validate arguments
    assert args.dataset, "Argument --dataset is required"
    
    print("Weights: ", args.weights)
    dataset = os.path.basename(os.path.normpath(args.dataset))
    dataset_train = os.path.join(args.dataset, 'train/')
    dataset_test = os.path.join(args.dataset, 'test/')
    print("Dataset: ", dataset)
    print("Logs: ", args.logs)

    fs = glob.glob(os.path.join(dataset_train, '*_img.tif'))
    ntrain = len(fs)
    nval = ntrain//8
    print('ntrain %d nval %d'%(ntrain, nval))
    # Configurations
    config = NucleusConfig()
    config.BATCH_SIZE = batch_size
    config.IMAGE_SHAPE = [256,256,3]
    config.IMAGES_PER_GPU = batch_size
    config.LEARNING_RATE = learning_rate
    config.HEAD_EPOCHS = 20
    config.TRAIN_EPOCHS = nepochs
    config.STEPS_PER_EPOCH = (ntrain - nval) // config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = max(1, nval // config.IMAGES_PER_GPU)
    #    else:
    #    config = NucleusInferenceConfig()
    
    
    config.display()

    # Give the configuration a recognizable name
    config.NAME = dataset
    

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # train model
    
    train(model, dataset_train)
    #pdb.set_trace()
    weights_path = model.checkpoint_path.format(epoch=model.epoch)
    print(weights_path)
    # reload model in inference mode
    config = NucleusInferenceConfig()
    config.NAME = dataset
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    
    model.load_weights(weights_path, by_name=True)
    overlapping_masks = detect(model, dataset_test)
    
    # score output
    #ndataset = NucleusDataset()
    #ndataset.load_nucleus(dataset_test, '')
    #ndataset.prepare()
    #APs = compute_batch_ap(ndataset, ndataset.image_ids)
    #print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))

    masks = []
    for i in range(len(overlapping_masks)):
        mask = overlapping_masks[i]
        medians = []
        for m in range(mask.shape[-1]):
            ypix, xpix = np.nonzero(mask[:,:,m])
            medians.append(np.array([ypix.mean(), xpix.mean()]))
        masks.append(np.int32(remove_overlaps(np.transpose(mask, (2,0,1)), 
                                                           mask.sum(axis=-1), np.array(medians))))
    mlist = glob.glob(os.path.join(dataset_test, '*_masks.tif'))
    Y_test = [skimage.io.imread(fimg)+1 for fimg in mlist]
    rez = matching.matching_dataset(Y_test, masks, thresh=[0.5,0.75,.9], by_image=True)
    print(rez)