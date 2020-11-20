"""
Mask R-CNN
Configurations and data loading code for FoodDataset.

------------------------------------------------------------

Usage: (see colab notebook for full explanation), or run from
       the command line as such:
    wandb init
    Inside MaskRCNN_wandb directory:
    # Train a new model starting from pre-trained COCO weights
    python samples/food/food.py train --dataset=/path/to/data-folder/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 samples/food/food.py train --dataset=/path/to/data-folder/ --model=imagenet 

    # Continue training a model that you had trained earlier
    python3 samples/food/food.py train --dataset=/path/to/data-folder/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 samples/food/food.py train --dataset=/path/to/data-folder/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 samples/food/food.py evaluate --dataset=/path/to/data-folder/ --model=last
"""

import os
import sys
import time
import numpy as np
import scipy
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)
import wandb
import matplotlib.pyplot as plt
import keras

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from mrcnn.dataset import FoodDataset
from mrcnn.evaluate import build_coco_results, evaluate_coco

from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
if os.path.exists(DEFAULT_LOGS_DIR)==False:
    os.mkdir(DEFAULT_LOGS_DIR)
############################################################
#  Configurations
############################################################


class FoodConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "food-challenge"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 5

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # annotations-small.json has 40 classes

    # Halve STEPS_PER_EPOCH to speed up training time for the sake of demonstration
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS=50

    IMAGE_MAX_DIM=256
    IMAGE_MIN_DIM=256

    # MODEL TUNING
    if os.environ.get('BACKBONE'):
        BACKBONE = os.environ.get('BACKBONE')
    if os.environ.get('GRADIENT_CLIP_NORM'):
        GRADIENT_CLIP_NORM = float(os.environ.get('GRADIENT_CLIP_NORM'))
    if os.environ.get('LEARNING_RATE'):
        LEARNING_RATE = float(os.environ.get('LEARNING_RATE'))
    if os.environ.get('WEIGHT_DECAY'):
        WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY'))
    

    def get_config_dict(self):
        """Return Configuration values as a dictionary for the sake of syncing with wandb"""
        d = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                d[a] = getattr(self, a)
        return d


############################################################
# WANDB
############################################################

run = wandb.init(project="foodr3-community")
_config = FoodConfig()

config_dict = _config.get_config_dict()
configs_of_interest = ['BACKBONE', 'GRADIENT_CLIP_NORM', 'LEARNING_MOMENTUM', 'LEARNING_RATE',
                        'WEIGHT_DECAY', 'STEPS_PER_EPOCH']

wandb.log({k: config_dict[k] for k in configs_of_interest})


def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf

class ImageCallback(keras.callbacks.Callback):
    def __init__(self, run, dataset_val, dataset_train,infer_config,log_dir):
        super(ImageCallback, self).__init__()
        self.run = run
        self.dataset_val = dataset_val
        self.dataset_train = dataset_train
        self.image_ids = dataset_val.image_ids[:3]
        self.infer_config = infer_config
        self.log_dir=log_dir
        self.inf_model = modellib.MaskRCNN(mode="inference", config=self.infer_config ,
                                  model_dir=log_dir,callbacks=[])
        
    def load_curr_model(self):
        model_path = self.inf_model.find_last()[1]
        self.inf_model.load_weights(model_path, by_name=True)

    def predict_image(self, image_id):
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            self.dataset_val, _config, image_id, use_mini_mask=False)
        _, ax = plt.subplots(figsize=(16, 16)) 
        
        # Run detection
        results = self.inf_model.detect([original_image])

        # Visualize results
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                self.dataset_val.class_names, r['scores'],figsize=(16,16), 
            ax=ax)
        return fig_to_array(ax.figure)

    def on_epoch_end(self, epoch, logs):
        print("Uploading images to wandb...")
        self.load_curr_model();
        predicted_images = [self.predict_image(i) for i in self.image_ids]
        wandb.log({"img_segmentations":[
            wandb.Image(
                scipy.misc.imresize(img, 50),
                caption="SampleImage",
                mode='RGBA') for img in predicted_images]})

class PerformanceCallback(keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
    def on_epoch_end(self, epoch, logs):
        print("Uploading metrics to wandb...")
        wandb.log(logs)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on FoodDataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on FoodDataset")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/FoodDataset/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FoodConfig()
    else:
        class InferenceConfig(FoodConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

     
    dataset_train = FoodDataset()
    dataset_val = FoodDataset()
    dataset_val.load_dataset(args.dataset+"\\val", load_small=True,return_coco=False)
    dataset_val.prepare()
    class InferenceConfig(FoodConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
    inf_config = InferenceConfig()
    callbacks = [
            ImageCallback(
                run,
                dataset_val,
                dataset_train,inf_config,args.logs),
            PerformanceCallback(run)
    ]

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs, callbacks=callbacks)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs, callbacks=[])

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.command =="train":
          model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        # Training dataset. 
        dataset_train.load_dataset(args.dataset+"\\train", True,False)
        dataset_train.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
        run.finish()

    elif args.command == "evaluate":
        print("Running COCO evaluation on val images.")
        coco = dataset_val.load_dataset(args.dataset+"\\val",load_small=True, return_coco=True)
        evaluate_coco(model, dataset_val, coco, "segm")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
