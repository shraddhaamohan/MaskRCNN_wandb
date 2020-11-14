## Intro

Mask R-CNN is a deep learning model for computer vision developed by the Facebook AI group that achieves state-of-the-art results on semantic segmentation (object recognition and pixel labeling) tasks. An implementation of the model is made available by Matterport on their [github page](github.com/matterport). The code in their repo works with MS Coco (a benchmark dataset for semantic segmentation) out of the box, but provides for easy extensibility to any kind of dataset or image segmentation task.

![Instance Segmentation Sample](assets/street.png)

This fork of the [matterport/mask_rcnn repo](github.com/matterport/mask-rcnn) was set up to integrate with Weights and Biases (wandb). wandb is a cloud interface for tracking model parameters and performance, allowing machine learning teams to coordinate work in a way similar to github.

We customised this repo to work with a custom dataset, particularly the Food Recognition Challenge Dataset for the competition hosted on AICrowd.

[Here](https://app.wandb.ai/trentwatson1/mask-rcnn) are the results of our runs.

For a more detailed overview of our process and results see our [discourse post](https://medium.com/@connorandtrent/mask-r-cnn-hyperparameter-experiments-with-weights-and-biases-bd2319faae26) or the
"What's on my plate? notebook present here (Runs on colab!).

## Setup

To run this experiment locally create a conda environment named tf-maskrcnn

1. `source activate tf_maskrcnn`
2. `git clone https://github.com/connorhough/mask_rcnn`
3. `cd mask_rcnn`
4. `pip install cython`
5. `pip install -r requirements.txt`
6. `python setup.py install`
7. `wandb init`, then follow the init steps
8. `python samples/food/food.py train --model=coco --dataset=path_to_data_dir`

The parameter sweep can be run with `./sweep.sh`
