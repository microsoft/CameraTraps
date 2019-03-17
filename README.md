# Overview

This repo contains the tools for training, running, and evaluating detectors and classifiers for images collected from motion-triggered camera traps.  The core functionality provided is:

- Data parsing from frequently-used camera trap metadata formats into a common format
- Training and evaluation of detectors, particularly our "megadetector", which does a pretty good job finding terrestrial animals in a variety of ecosystems
- Training and evaluation of species-level classifiers for specific data sets
- A Web-based demo that runs our models via a REST API that hosts them on a Web endpoint
- Miscellaneous useful tools for manipulating camera trap data
- Research experiments we're doing around camera trap data (i.e., some directories are highly experimental and you should take them with a grain of salt)

This repo is maintained by folks in the [Microsoft AI for Earth](http://aka.ms/aiforearth) program who like looking at pictures of animals.  I mean, we want to use machine learning to support conservation too, but we also really like looking at pictures of animals.


# Data

This repo does not directly host camera trap data, but we work with our collaborators to make data and annotations available whenever possible on [lila.science](http://lila.science).


# Models

This repo does not extensively host models, though we will release models when they are at a level of generality that they might be useful to other people.  


## MegaDetector

Speaking of models that might be useful to other people, we have trained a one-class animal detector trained on several hundred thousand bounding boxes from a variety of ecosystems.  The model is trained with the TensorFlow Object Detection API; it can be downloaded [here](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.pb) (.pb).  We use this as our first stage for classifier training and inference.  An example script for invoking this detector on new images can be found [here](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py).  Let us know how it works on your images!


# Contact

For questions about this repo, contact [cameratraps@microsoft.com](mailto:cameratraps@microsoft.com).


# Contents

This repo is organized into the following folders...

## annotations

Code for creating new annotation tasks and converting annotations to COCO-CameraTraps format.


## classification

Code for training species classifiers on new data sets, generally trained on crops generated via an existing detector.


## database_tools

Code for creating, visualizing stats, or editing COCO-CameraTraps style json databases.

## demo

Source for the Web-based demo that runs our models via a REST API that hosts them on a Web endpoint.


## detection

Code for training and evaluating detectors.


## experiments

Ongoing research projects that use this repository in one way or another; as of the time I'm editing this README, there are projects in this folder around active learning and the use of simulated environments for training data augmentation.


## sandbox

Random things that don't fit in any other directory.  Currently contains a single file, a not-super-useful but super-duper-satisfying and mostly-successful attempt to use OCR to pull metadata out of image pixels in a fairly generic way, to handle those pesky cases when image metadata is lost.


## tfrecords

Code for creating or reading from tfrecord files, based on https://github.com/visipedia/tfrecords .

# Gratuitous pretty camera trap picture

![alt text](http://lila.science/wp-content/uploads/2018/10/IMG_1881_web.jpg "Bird flying above water")

Image credit USDA, from the [NACTI](http://lila.science/datasets/nacti) data set.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [cla.microsoft.com](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

