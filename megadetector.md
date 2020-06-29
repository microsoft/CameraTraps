## Table of contents

1. [MegaDetector overview](#megadetector-overview)
2. [Our ask to MegaDetector users](#our-ask-to-megadetector-users)
3. [Downloading the model(s)](#downloading-the-models)
4. [Using the models](#using-the-models)
5. [Tell me more about why detectors are a good first step for camera trap images](#tell-me-more-about-why-detectors-are-a-good-first-step-for-camera-trap-images)
6. [Pretty picture](#pretty-picture)
7. [Mesmerizing video](#mesmerizing-video)
8. [Can you share the training data?](#can-you-share-the-training-data)


## MegaDetector overview

Conservation biologists invest a huge amount of time reviewing camera trap images, and &ndash; even worse &ndash; a huge fraction of that time is spent reviewing images they aren't interested in.  This primarily includes empty images, but for many projects, images of people and vehicles are also "noise", or at least need to be handled separately from animals.

*Machine learning can accelerate this process, letting biologists spend their time on the images that matter.*

To this end, this page hosts a model we've trained to detect animals, people, and vehicles in camera trap images, using several hundred thousand bounding boxes from a variety of ecosystems.  It does not identify animals, it just finds them.  The current model is based on Faster-RCNN with an InceptionResNetv2 base network, and was trained with the TensorFlow Object Detection API.  We use this model as our first stage for classifier training and inference.

## Our ask to MegaDetector users

MegaDetector is free, and it makes us super-happy when people use it, so we put it out there as a downloadable model that is easy to use in a variety of conservation scenarios.  That means we don't know who's using it unless you contact us (or we happen to run into you), so please please pretty-please email us at [cameratraps@microsoft.com](mailto:cameratraps@microsoft.com) if you find it useful!


## Downloading the model(s)

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


## Using the models

We provide three ways to apply this model to new images:

1. A simple test script that makes neat pictures with bounding boxes, but doesn't produce a useful output file ([run_tf_detector.py](https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector.py)) 
2. A script for running large batches of images on a local GPU ([run_tf_detector_batch.py](https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector_batch.py)) 
3. A [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/postprocessing) that runs images on many GPUs at once on Azure

This section describes how to run the two scripts, including installing all the necessary Python dependencies. If your computer is also used for other Python projects, we recommend you set up the environment as described in the [Installation](https://github.com/microsoft/CameraTraps#installation) section of our main README, and use conda to set up a virtual environment in which to run scripts from this repo. This reduces potential version conflict headaches with your other projects. The environment file you should use to run the two scripts below is `environment-detector.yml`. You will still need to add the required repos to `PYTHONPATH`, but don't have to worry about installing Python, pip or any packages yourself. If you do not have a GPU on your computer, change `tensorflow-gpu` to `tensorflow` in `environment-detector.yml`.

### 0. prerequisites

When we describe how to run our two inference scripts below, we assume the following:

1. You have Python 3 installed.  We recommend installing [Anaconda](https://www.anaconda.com/products/individual), which is Python plus a zillion useful packages.
2. You have checked out this git repo, and the [AI for Earth Utilities](http://github.com/microsoft/ai4eutils) repo.  If you're not familiar with git and are on a Windows machine, we recommend installing [Git for Windows](https://git-scm.com/download/win).  Specific instructions for checking out the repo will be rolled into the next step.
3. You have added both directories where you cloned the two repos to your PYTHONPATH environment variable.  Here's a [good page](https://www.computerhope.com/issues/ch000549.htm) about editing environment variables in Windows.  You will need administrative access to your PC to set an environment variable.

Here are instructions for steps 2 and 3 that assume you *don't* have administrative access to your PC (if you set the environment variable as per above, you can skip the "set PYTHONPATH" step here).  We're going to clone the repos to "c:\git", but you can use any folder you like.

After installing git and Anaconda, open an Anaconda Prompt, and run:

```batch
mkdir c:\git
cd c:\git
git clone https://github.com/Microsoft/cameratraps
git clone https://github.com/Microsoft/ai4eutils
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
```

On subsequent times you open your Anaconda prompt, you just need to do:

```batch
cd c:\git\cameratraps\api\batch_processing\postprocessing
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
```

### 1. run_tf_detector.py

To "test" this model on small sets of images and get super-satisfying visual output, we provide [run_tf_detector.py](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py), an example script for invoking this detector on new images.  This isn't how we recommend running lots of images through MegaDetector (see [run_tf_detector_batch.py](#2-run_tf_detector_batchpy) below for "real" usage), but it's a quick way to test things out.  [Let us know](mailto:cameratraps@microsoft.com) how it works on your images!

#### Running run_tf_detector.py on Linux

To try this out (on Linux), assuming you have Python 3 and pip installed, you can run the following:

```bash
# Download the script and the MegaDetector model file
wget https://raw.githubusercontent.com/microsoft/CameraTraps/master/detection/run_tf_detector.py
wget https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# Install TensorFlow
#
# If you have a GPU on your computer, change "tensorflow" to "tensorflow-gpu"
pip install tensorflow==1.13.1

# Install other dependencies
pip install Pillow humanfriendly matplotlib tqdm jsonpickle

# Run MegaDetector
python run_tf_detector.py md_v4.1.0.pb --image_file some_image_file.jpg
```

Run `python run_tf_detector.py` for a full list of options.

#### Running run_tf_detector.py on Windows

The "git", "pip", and "Python" lines above will work fine for Windows if you have Anaconda installed, and you will probably want to download the [script](https://raw.githubusercontent.com/microsoft/CameraTraps/master/detection/run_tf_detector.py) and the [MegaDetector model file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb) through your browser.

Then you can do the following, changing "tensorflow" to "tensorflow-gpu" if you have a GPU on your computer:

```batch
pip install tensorflow==1.13.1
pip install Pillow humanfriendly matplotlib tqdm
python where_you_downloaded_the_script/run_tf_detector.py where_you_downloaded_the_detector_file/md_v4.1.0.pb --image_file some_image_file.jpg
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
pip install humanfriendly Pillow pandas tqdm

# Run MegaDetector
python CameraTraps/detection/run_tf_detector_batch.py ~/md_v4.1.0.pb some_image_file.jpg some_output_file.json
```

Run `python run_tf_detector_batch.py` for a full list of options.

#### Running run_tf_detector_batch.py on Windows

Once you've gone through the [prerequisites](#0-prerequisites) above, all the `git`, `pip`, and `python` steps in the above instructions should work fine on Windows too.  In the Linux instructions above, we suggest using `wget` to download the model file; on Windows, you'll probably want to download the [MegaDetector model file](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb) in your browser.

Putting together the prerequisites, package installation, and execution steps for Windows, once you've installed Anaconda, this might look like the following (in your Anaconda prompt):

```cd c:\git
git clone https://github.com/microsoft/CameraTraps/
git clone https://github.com/microsoft/ai4eutils/
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
pip install tensorflow==1.13.1
pip install humanfriendly Pillow pandas tqdm
python CameraTraps/detection/run_tf_detector_batch.py wherever_you_put_the_detector_file/md_v4.1.0.pb some_image_file.jpg some_output_file.json
```


### 3. Batch processing API

Speaking of lots of images, when we process loads of images from collaborators, we use our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), which we can make available externally on request.  [Email us](mailto:cameratraps@microsoft.com) for more information.


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
