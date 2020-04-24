### Commit histories

    1st commit - Folder structure setup i.e submodule added and empty files created.
    2nd commit - Environment called `efficiendet` is setup with required packages and exported as `environment-efficient.yml`. Ran training and inference script with a sample dataset. Used `Python-3.7` in the environment. Works fine with `Python-3.6` too. PyCocoTools had issues with Python 3.8 and numpy.

#### install requirements
    pip install numpy Cython
    pip install pycocotools opencv-python tqdm tensorboard tensorboardX pyyaml matplotlib
    pip install torch==1.4.0
    pip install torchvision==0.5.0

OR

    conda env create --file environment-efficient.yml

#### download and unzip dataset
    mkdir datasets
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.1/dataset_shape.tar.gz -O datasets/dataset_shape.tar.gz
    tar xzf datasets/dataset_shape.tar.gz

#### download pretrained weights
    mkdir weights
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth

#### prepare project file projects/shape.yml
    cat projects/shape.yml

#### run the simple inference script
    python efficientdet_test.py

### ToDo: ->

    1. Understand what different weights like efficientdet-d0,efficientdet-d1 stand for. Revise obj-detection concepts.
    2. Get dataset from `marmot` and run training script.