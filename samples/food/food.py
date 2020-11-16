"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
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
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # annotations-small.json has 12 classes

    # Halve STEPS_PER_EPOCH to speed up training time for the sake of demonstration
    STEPS_PER_EPOCH = 750
    USE_MINI_MASK=False
    LEARNING_RATE=0.0001


    BACKBONE = "resnet50"
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
        #self.run.history.row.update(logs)
        wandb.log(logs)
        #self.run.history.add()

############################################################
#  Dataset
############################################################

class FoodDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_small=False, return_coco=True):
        """ Loads dataset released for the AICrowd Food Challenge
            Params:
                - dataset_dir : root directory of the dataset (can point to the train/val folder)
                - load_small : Boolean value which signals if the annotations for all the images need to be loaded into the memory,
                               or if only a small subset of the same should be loaded into memory
        """
        self.load_small = load_small
        if self.load_small:
            annotation_path = os.path.join(dataset_dir, "annotations-small.json")
        else:
            annotation_path = os.path.join(dataset_dir, "annotations.json")

        image_dir = os.path.join(dataset_dir, "images")
        print("Annotation Path ", annotation_path)
        print("Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        classIds = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for _class_id in classIds:
            self.add_class(
                "food-challenge",
                _class_id,
                self.coco.loadCats(_class_id)[0]["name"],
            )

        # Register Images
        for _img_id in image_ids:
            assert os.path.exists(
                os.path.join(image_dir, self.coco.imgs[_img_id]["file_name"])
            )
            self.add_image(
                "food-challenge",
                image_id=_img_id,
                path=os.path.join(image_dir, self.coco.imgs[_img_id]["file_name"]),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(
                    self.coco.getAnnIds(imgIds=[_img_id], catIds=classIds, iscrowd=None)
                ),
            )

        if return_coco:
            return self.coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        
        image_info = self.image_info[image_id]
        assert image_info["source"] == "food-challenge"

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "food-challenge.{}".format(annotation["category_id"])
            )
            if class_id:
                m = self.annToMask(
                    annotation, image_info["height"], image_info["width"]
                )
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FoodDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a reference for a particular image

            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return "food-challenge::{}".format(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "food-challenge"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="segm", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


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
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train.load_dataset(args.dataset+"\\train", True,False)
        dataset_train.prepare()


        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=3,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=4,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=5,
                    layers='all',
                    augmentation=augmentation)
        run.finish()

    elif args.command == "evaluate":
        print("Running COCO evaluation on val images.")
        coco = dataset_val.load_dataset(args.dataset+"/val",load_small=True, return_coco=True)
        evaluate_coco(model, dataset_val, coco, "segm")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
