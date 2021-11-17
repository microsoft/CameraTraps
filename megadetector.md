## Table of contents

1. [MegaDetector overview](#megadetector-overview)
2. [Our ask to MegaDetector users](#our-ask-to-megadetector-users)
3. [Who is using MegaDetector?](#who-is-using-megadetector)
4. [How fast is MegaDetector, and can I run it on my giant/small computer?](#how-fast-is-megadetector-and-can-i-run-it-on-my-giantsmall-computer)
5. [Downloading the model](#downloading-the-model)
6. [Using the model](#using-the-model)
7. [Tell me more about why detectors are a good first step for camera trap images](#tell-me-more-about-why-detectors-are-a-good-first-step-for-camera-trap-images)
8. [Pretty picture](#pretty-picture)
9. [Mesmerizing video](#mesmerizing-video)
10. [Can you share the training data?](#can-you-share-the-training-data)


## MegaDetector overview

Conservation biologists invest a huge amount of time reviewing camera trap images, and &ndash; even worse &ndash; a huge fraction of that time is spent reviewing images they aren't interested in.  This primarily includes empty images, but for many projects, images of people and vehicles are also "noise", or at least need to be handled separately from animals.

*Machine learning can accelerate this process, letting biologists spend their time on the images that matter.*

To this end, this page hosts a model we've trained - called "MegaDetector" - to detect animals, people, and vehicles in camera trap images.  It does not identify animals, it just finds them.  

This page is about the technical elements of MegaDetector; if you are an ecologist looking to use MegaDetector, you may prefer to start at our [MegaDetector collaborations page](collaborations.md).

The current model is based on Faster-RCNN with an InceptionResNetv2 base network, and was trained with the TensorFlow Object Detection API, using several hundred thousand bounding boxes from a variety of ecosystems.


## Our ask to MegaDetector users

MegaDetector is free, and it makes us super-happy when people use it, so we put it out there as a downloadable model that is easy to use in a variety of conservation scenarios.  That means we don't know who's using it unless you contact us (or we happen to run into you), so please please pretty-please email us at [cameratraps@lila.science](mailto:cameratraps@lila.science) if you find it useful!


## How fast is MegaDetector, and can I run it on my giant/small computer?

We often run MegaDetector on behalf of users as a free service; see our [MegaDetector collaborations page](collaborations.md) for more information.  But there are many reasons to run MegaDetctor on your own, and how practical this is will depend in part on how many imags you need to process and what kind of computer hardware you have available.  MegaDetector is designed to favor accuracy over speed, and we typically run it on <a href="https://en.wikipedia.org/wiki/Graphics_processing_unit">GPU</a>-enabled computers.  That said, you can run anything on anything if you have enough time, and we're happy to support users who run MegaDetector on their own GPUs (in the cloud or on their own PCs), on their own CPUs, or even on embedded devices.  If you only need to process a few thousand images per week, for example, a typical laptop will be just fine.  If you want to crunch through 20 million images as fast as possible, you'll want at least one GPU.

Here are some good rules of thumb to help you estimate how fast you can run MegaDetector on different types of hardware:

* On a decent laptop (without a fancy deep learning GPU) that is neither the fastest nor slowest laptop you can buy in 2021, MegaDetector takes somewhere between eight and twenty seconds per image, depending on how many CPUs you use.  This works out to being able to process somewhere between 4,000 and 10,000 image per day.  This might be totally fine for scenarios where you have even hundreds of thousands of images, as long as you can wait a few days.
* On a dedicated deep learning GPU that is neither the fastest nor slowest GPU you can buy in 2021, MegaDetector takes between 0.3 and 0.5 seconds per image, which works out to between 200,000 and 250,000 images per day.  We also include a few <a href="#benchmark-timings">benchmark timings</a> below on some specific GPUs.

We don't typically recommend running MegaDetctor on embedded devices, although <a href="https://www.electromaker.io/project/view/whats-destroying-my-yard-pest-detection-with-raspberry-pi">some folks have done it</a>!  More commonly, for embedded scenarios, it probably makes sense to use MegaDetector to generate bounding boxes on lots of images from your specific ecosystem, then use those boxes to train a smaller model that fits your embedded device's compute budget.

### Benchmark timings

We haven't done a lot of measurement on how long MegaDetector takes to run on "typical" GPUs, because we always use the same GPUs (we typically use 16 NVIDIA V100 GPUs to run large image batches in the cloud).  But we would like to start tracking this to help users make decisions about buying GPUs, so if you are using MegaDetector and have a GPU you're willing to benchmark on, <a href="mailto:cameratraps@lila.science">email us</a>!

But with a test batch of around 13,000 images from the public <a href="https://lila.science/datasets/snapshot-karoo">Snapshot Karoo</a> and <a href="http://lila.science/datasets/idaho-camera-traps/">Idaho Camera Traps</a> datasets:

* An <a href="https://www.nvidia.com/en-us/data-center/v100/">NVIDIA V100</a> GPU processes around 2.79 images per second, or around 240,000 images per day
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/">NVIDIA RTX 3090</a> GPU processes around 3.24 images per second, or around 280,000 images per day
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/">NVIDIA RTX 2080 Ti</a> GPU processes around 2.48 images per second, or around 214,000 images per day
* An <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2060-super/">NVIDIA RTX 2060 SUPER</a> GPU processes around 1.64 images per second, or around 141,000 images per day
* An <a href="https://www.nvidia.com/en-us/titan/titan-v/">NVIDIA Titan V</a> GPU processes around 1.9 images per second, or around 167,000 images per day

If you want to run this benchmark on your own, here are <a href="download_megadetector_timing_benchmark_set.bat">azcopy commands</a> to download those 13,226 images, and we're happy to help you get MegaDetector running on your setup.  Or if you're using MegaDetector on other images with other GPUs, we'd love to include that data here as well.  <a href="mailto:cameratraps@lila.science">Email us</a>!


## Who is using MegaDetector?

See <a href="https://github.com/microsoft/CameraTraps/#who-is-using-the-ai-for-earth-camera-trap-tools">this list</a> on the repo's main page.


## Downloading the model

### MegaDetector v4.1, 2020.04.27

#### Release notes

This release incorporates additional training data from Borneo, Australia and the [WCS Camera Traps](http://lila.science/datasets/wcscameratraps) dataset, as well as images of humans in both daytime and nighttime. We also have a preliminary "vehicle" class for cars, trucks and bicycles.

#### Download links

- [Frozen model (.pb)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb)
- [TFODAPI config file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.config)
- [Last checkpoint (for resuming training)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_checkpoint.zip)
- [Tensorflow SavedModel for TFServing](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_saved_model.zip) (inputs in uint8 format, `serving_default` output signature)

### MegaDetector v3, 2019.05.30

#### Release notes

In addition to incorporating additional data, this release adds a preliminary "human" class.  Our animal training data is still far more comprehensive than our humans-in-camera-traps data, so if you're interested in using our detector but find that it works better on animals than people, stay tuned.

#### Download links

- [Frozen model (.pb)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb)
- [TFODAPI config file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.config)
- [Last checkpoint (for resuming training)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3_checkpoint.zip)
- [TensorFlow SavedModel](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_normalized_megadetector_v3_tf19.tar.gz) (inputs in TF [common image format](https://www.tensorflow.org/hub/common_signatures/images#image_input), `default` output signature)
- [Tensorflow SavedModel for TFServing](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip) (inputs in uint8 format, `serving_default` output signature)

### MegaDetector v2, 2018

#### Release notes

First MegaDetector release!

#### Download links

- [Frozen model (.pb)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.pb)
- [TFODAPI config file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2.config)
- [Last checkpoint (for resuming training)](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v2_checkpoint.zip)


## Using the model

We provide four ways to apply this model to new images:

1. Try applying the MegaDetector to some of your own images in Google Drive using this [Colab notebook](https://github.com/microsoft/CameraTraps/blob/master/detection/megadetector_colab.ipynb).

<a href="https://colab.research.google.com/github/microsoft/CameraTraps/blob/master/detection/megadetector_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>

2. A simple test script that makes neat pictures with bounding boxes, but doesn't produce a useful output file ([run_tf_detector.py](https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector.py)) 
3. A script for running large batches of images on a local GPU ([run_tf_detector_batch.py](https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector_batch.py)) 
4. A [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing) that runs images on many GPUs at once on Azure

This section describes how to run the two scripts (options 2 and 3), including installing all the necessary Python dependencies. If your computer is also used for other Python projects, we recommend you set up the environment as described in the [Installation](https://github.com/microsoft/CameraTraps#installation) section of our main README, and use conda to set up a virtual environment in which to run scripts from this repo. This reduces potential version conflict headaches with your other projects. The environment file you should use to run the two scripts below is `environment-detector.yml`. You will still need to add the required repos to `PYTHONPATH`, but don't have to worry about installing Python, pip or any packages yourself. If you do not have a GPU on your computer, change `tensorflow-gpu` to `tensorflow` in `environment-detector.yml`.

### 0. prerequisites

When we describe how to run our two inference scripts below, we assume the following:

1. You have Python 3 installed.  We recommend installing [Anaconda](https://www.anaconda.com/products/individual), which is Python plus a zillion useful packages.
2. You have downloaded our [MegaDetector model](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb) to some location on your computer.
3. You have cloned this git repo, and the [AI for Earth Utilities](http://github.com/microsoft/ai4eutils) repo.  If you're not familiar with git and are on a Windows machine, we recommend installing [Git for Windows](https://git-scm.com/download/win).  Specific instructions for checking out the repo will be rolled into the next step.
4. You have added both directories where you cloned the two repos to your PYTHONPATH environment variable.  Here's a [good page](https://www.computerhope.com/issues/ch000549.htm) about editing environment variables in Windows.  You will need administrative access to your PC to set an environment variable.

Here are Windows instructions for steps 3 and 4 that assume you *don't* have administrative access to your PC (if you set the environment variable as per above, you can skip the "set PYTHONPATH" step here).  We're going to clone the repos to "c:\git", but you can use any folder you like.

After installing git and Anaconda, open an Anaconda Prompt, and run:

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/Microsoft/cameratraps -b tf1-compat
git clone https://github.com/Microsoft/ai4eutils
pip install tensorflow pillow humanfriendly matplotlib tqdm jsonpickle statistics requests
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
```

On subsequent times you open your Anaconda prompt, you just need to do:

```batch
cd c:\git\cameratraps\api\batch_processing\postprocessing
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
```

### 1. run_tf_detector.py

To "test" this model on small sets of images and get super-satisfying visual output, we provide [run_tf_detector.py](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py), an example script for invoking this detector on new images.  This isn't how we recommend running lots of images through MegaDetector (see [run_tf_detector_batch.py](#2-run_tf_detector_batchpy) below for "real" usage), but it's a quick way to test things out.  [Let us know](mailto:cameratraps@lila.science) how it works on your images!

#### Running run_tf_detector.py on Linux

To try this out (on Linux), assuming you have Python 3 and pip installed, you can run the following:

```bash
# Download the script and the MegaDetector model file
wget https://raw.githubusercontent.com/microsoft/CameraTraps/master/detection/run_tf_detector.py
wget https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# Install dependencies
pip install tensorflow pillow humanfriendly matplotlib tqdm jsonpickle statistics requests

# Run MegaDetector
python run_tf_detector.py md_v4.1.0.pb --image_file some_image_file.jpg
```

Run `python run_tf_detector.py` for a full list of options.

#### Running run_tf_detector.py on Windows

This assumes you've run the [prerequisites](#0-prerequisites) steps above.  After that, you can run the following in your Anaconda prompt:

```batch
python c:\git\CameraTraps\detection\run_tf_detector.py c:\wherever\you\downloaded\the\detector\file\md_v4.1.0.pb --image_file some_image_file.jpg
```


### 2. run_tf_detector_batch.py

To apply this model to larger image sets on a single machine, we recommend a slightly different script, [run_tf_detector_batch.py](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector_batch.py).  This outputs data in the same format as our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), so you can leverage all of our post-processing tools.

#### Running run_tf_detector_batch.py on Linux

To try this out (on Linux), assuming you have Python 3 and pip installed, you can do:

```bash
# Clone our two required git repos
git clone https://github.com/microsoft/CameraTraps/
git clone https://github.com/microsoft/ai4eutils/

# Add those repos to your Python path
export PYTHONPATH="$PYTHONPATH:$PWD/ai4eutils:$PWD/CameraTraps"

# Download the MegaDetector model file
wget -O ~/md_v4.1.0.pb https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# Install TensorFlow
#
# If you have a GPU on your computer, change "tensorflow" to "tensorflow-gpu"
pip install tensorflow==1.13.1

# Install other dependencies
pip install pandas tqdm pillow humanfriendly matplotlib tqdm jsonpickle statistics requests

# Run MegaDetector
python CameraTraps/detection/run_tf_detector_batch.py ~/md_v4.1.0.pb some_image_file.jpg some_output_file.json
```

Run `python run_tf_detector_batch.py` for a full list of options.

#### Running run_tf_detector_batch.py on Windows

This assumes you've run the [prerequisites](#0-prerequisites) steps above.  After that, you can run the following in your Anaconda prompt:

```batch
python c:\git\CameraTraps\detection\run_tf_detector_batch.py c:\wherever\you\downloaded\the\detector\file\md_v4.1.0.pb some_image_folder some_output_file.json --output_relative_filenames --recursive
```


### 3. Batch processing API

Speaking of lots of images, when we process loads of images from collaborators, we use our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), which we can make available externally on request.  [Email us](mailto:cameratraps@lila.science) for more information.


## Citing MegaDetector

If you use the MegaDetector in a publication, please cite:
```BibTeX
@article{beery2019efficient,
  title={Efficient Pipeline for Camera Trap Image Review},
  author={Beery, Sara and Morris, Dan and Yang, Siyu},
  journal={arXiv preprint arXiv:1907.06772},
  year={2019}
}
```


## Tell me more about why detectors are a good first step for camera trap images

Can do!  See these [slides](http://dmorris.net/misc/cameratraps/ai4e_camera_traps_overview).

## Pretty picture

Here's a "teaser" image of what detector output looks like:

![alt text](images/detector_example.jpg "Red bounding box on fox")

Image credit University of Washington.


## Mesmerizing video

Here's a neat [video](http://dolphinvm.westus2.cloudapp.azure.com/video/detector_video.html) of our v2 detector running in a variety of ecosystems, on locations unseen during training.

<a href="http://dolphinvm.westus2.cloudapp.azure.com/video/detector_video.html">
<img width=600 src="http://dolphinvm.westus2.cloudapp.azure.com/video/mvideo.jpg">
</a>

Image credit [eMammal](https://emammal.si.edu/).


## Can you share the training data?

This model is trained on bounding boxes from a variety of ecosystems, and many of the images we use in training are not publicly-shareable for license reasons.  We do train in part on bounding boxes from two public data sets:

- [Caltech Camera Traps](http://lila.science/datasets/caltech-camera-traps)
- [Snapshot Serengeti](http://lila.science/datasets/snapshot-serengeti)

...so if our detector performs really well on those data sets, that's great, but it's a little bit cheating, because we haven't published the set of locations from those data sets that we use during training.
