## Table of contents

1. [MegaDetector overview](#megadetector-overview)<br/>
2. [Our ask to MegaDetector users](#our-ask-to-megadetector-users)<br/>
3. [Downloading the model(s)](#downloading-the-models)<br/>
4. [Using the models](#using-the-models)<br/>
5. [Tell me more about why detectors are a good first step for camera trap images](#tell-me-more-about-why-detectors-are-a-good-first-step-for-camera-trap-images)<br/>
6. [Pretty picture](#pretty-picture)<br/>
7. [Mesmerizing video](#mesmerizing-video)<br/>
8. [Can you share the training data?](#can-you-share-the-training-data)<br/>

## MegaDetector overview

Conservation biologists invest a huge amount of time reviewing camera trap images, and &ndash; even worse &ndash; a huge fraction of that time is spent reviewing images they aren&rsquo;t interested in.  This primarily includes empty images, but for many projects, images of people and vehicles are also &ldquo;noise&rdquo;, or at least need to be handled separately from animals.

<i>Machine learning can accelerate this process, letting biologists spend their time on the images that matter.</i>

To this end, this page hosts a model we&rsquo;ve trained to detect animals, people, and vehicles in camera trap images, using several hundred thousand bounding boxes from a variety of ecosystems.  It does not identify animals, it just finds them.  The current model is based on Faster-RCNN with an InceptionResNetv2 base network, and was trained with the TensorFlow Object Detection API.  We use this model as our first stage for classifier training and inference.

## Our ask to MegaDetector users

MegaDetector is free, and it makes us super-happy when people use it, so we put it out there as a downloadable model that is easy to use in a variety of conservation scenarios.  That means we don&rsquo;t know who&rsquo;s using it unless you contact us (or we happen to run into you), so please please pretty-please email us at <a href="mailto:cameratraps@microsoft.com">cameratraps@microsoft.com</a> if you find it useful!


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

In addition to incorporating additional data, this release adds a preliminary &ldquo;human&rdquo; class.  Our animal training data is still far more comprehensive than our humans-in-camera-traps data, so if you&rsquo;re interested in using our detector but find that it works better on animals than people, stay tuned.

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

We provide three ways to apply this model to new images - see below. For the two scripts, we describe how you can get set up if you have Python 3 and pip installed on a Linux computer or if you are on Windows and are starting from scratch. If your computer is also used for other Python projects, we recommend you set up the environment as described in the [Installation](https://github.com/microsoft/CameraTraps#installation) section of our main readme, and use conda to set up a virtual environment in which to run scripts from this repo. This reduces potential version conflict headaches with your other projects. The environment file you should use to run the two scripts below is `environment-detector.yml`. You will still need to add the required repos to `PYTHONPATH`, but don't have to worry about installing Python, pip or any packages yourself. If you do not have a GPU on your computer, change `tensorflow-gpu` to `tensorflow` in `environment-detector.yml`.

### 0. prerequisites

When we describe how to run our two inference scripts below, we assume the following:

1. You have Python installed.  If you&rsquo;re working on Windows, we recommend installing <a href="anaconda.com/distribution/">Anaconda</a>, which is Python plus a zillion useful packages.
2. You have checked out this git repo, and the <a href="http://github.com/microsoft/ai4eutils">AI for Earth Utilities</a> repo.  If you&rsquo;re not familiar with git, we recommend installing <a href="https://git-scm.com/download/win">Git for Windows</a>.  Specific instructions for checking out the repo will be rolled into the next step.
3. You have put the base folder of both repos on your Python path.  If you are using Windows, for example, you would do this by finding the directory to which you cloned each repo, and adding that directory to your PYTHONPATH environment variable.  Here&rsquo;s a <a href="https://www.computerhope.com/issues/ch000549.htm">good page</a> about editing environment variables in Windows.  You will need administrative access to your PC to set an environment variable.

Here are instructions for steps 2 and 3 that assume you <i>don&rsquo;t</i> have administrative access to your PC (if you set the environment variable as per above, you can skip the &ldquo;set PYTHONPATH&rdquo; step here).  We&rsquo;re going to put things in &ldquo;c:\git&rdquo;, but you can use any folder you like. 

After installing git and Anaconda, open an Anaconda Prompt, and run:

```mkdir c:\git
cd c:\git
git clone https://github.com/Microsoft/cameratraps
git clone https://github.com/Microsoft/ai4eutils
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
```

Subsequent times you open your Anaconda prompt, you'll just need to do:

```cd c:\git\cameratraps\api\batch_processing\postprocessing
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
python find_repeat_detections.py
```

### 1. run_tf_detector.py

To &ldquo;test drive&rdquo; this model on small sets of images and get super-satisfying visual output, we provide [run_tf_detector.py](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector.py), an example script for invoking this detector on new images.  This script doesn&rsquo;t depend on anything else in our repo, so you can download it and give it a try.  [Let us know](mailto:cameratraps@microsoft.com) how it works on your images!

#### Running run_tf_detector.py on Linux

To try this out (on Linux), assuming you have Python 3 and pip installed, you can do:

```
# Download the script and the MegaDetector model file
wget https://raw.githubusercontent.com/microsoft/CameraTraps/master/detection/run_tf_detector.py
wget https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb

# Install TensorFlow
#
# If you have a GPU on your computer, change "tensorflow" to "tensorflow-gpu"
pip install tensorflow==1.13.1

# Install other dependencies
pip install Pillow humanfriendly matplotlib tqdm jsonpickle

# Run MegaDetector
python run_tf_detector.py megadetector_v3.pb --image_file some_image_file.jpg
```

Run `python run_tf_detector.py` for a full list of options.

#### Running run_tf_detector.py on Windows

If you&rsquo;re using Windows, you&rsquo;ll need to install Python; if you&rsquo;re starting from scratch, we recommend installing <a href="https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/postprocessing/anaconda.com/distribution">Anaconda</a>, which is Python plus a zillion useful packages.  The &ldquo;git&rdquo;, &ldquo;pip&rdquo;, and &rdquo;Python&rdquo; lines above will then work fine for Windows too, and you will probably want to download the <a href="https://raw.githubusercontent.com/microsoft/CameraTraps/master/detection/run_tf_detector.py">script</a> and the <a href="https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb">MegaDetector model file</a> through your browser.

Then you can do the following, changing "tensorflow" to "tensorflow-gpu" if you have a GPU on your computer:

```
pip install tensorflow==1.13.1
pip install Pillow humanfriendly matplotlib tqdm
python wherever_you_downloaded_the_script/run_tf_detector.py wherever_you_downloaded_the_detector_file/megadetector_v3.pb --image_file some_image_file.jpg
```


### 2. run_tf_detector_batch.py

To apply this model to larger image sets on a single machine, we recommend a slightly different script, [run_tf_detector_batch.py](https://github.com/Microsoft/CameraTraps/blob/master/detection/run_tf_detector_batch.py).  This outputs data in the same format as our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), so you can leverage all of our post-processing tools.

#### Running run_tf_detector_batch.py on Linux

To try this out (on Linux), assuming you have Python 3 and pip installed, you can do:

```
# Clone our two required git repos
git clone https://github.com/microsoft/CameraTraps/
git clone https://github.com/microsoft/ai4eutils/

# Add those repos to your Python path
export PYTHONPATH="$PYTHONPATH:$PWD/ai4eutils:$PWD/CameraTraps"

# Download the MegaDetector model file
wget -O ~/megadetector_v3.pb https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb

# Install TensorFlow
#
# If you have a GPU on your computer, change "tensorflow" to "tensorflow-gpu"
pip install tensorflow==1.13.1

# Install other dependencies
pip install humanfriendly Pillow pandas tqdm

# Run MegaDetector
python CameraTraps/detection/run_tf_detector_batch.py ~/megadetector_v3.pb some_image_file.jpg some_output_file.json
```

Run `python run_tf_detector_batch.py` for a full list of options.

#### Running run_tf_detector_batch.py on Windows

If you&rsquo;re using Windows:

* You&rsquo;ll need to install Python; if you&rsquo;re starting from scratch, we recommend installing <a href="https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/postprocessing/anaconda.com/distribution">Anaconda</a>, which is Python plus a zillion useful packages.
* You&rsquo;ll need to install git; if you don&rsquo;t have git installed, we recommend installing <a href="https://git-scm.com/download/win">Git for Windows</a>.

Then all the `git`, `pip`, and `python` steps in the above instructions should work fine on Windows.  You&rsquo;ll probably want to just download the <a href="https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb">MegaDetector model file</a> in your browser.  For the step where you add our repos to your Python path, you can do this by finding the directory to which you cloned each repo, and adding that directory to your PYTHONPATH environment variable. Here’s a <a href="https://www.computerhope.com/issues/ch000549.htm">good page</a> about editing environment variables in Windows.  You can also do all of this at the command prompt.  

Putting all of this together, if you&rsquo;ve installed Anaconda, this might look like the following (in your Anaconda prompt):

```
cd c:\git
git clone https://github.com/microsoft/CameraTraps/
git clone https://github.com/microsoft/ai4eutils/
set PYTHONPATH=c:\git\cameratraps;c:\git\ai4eutils
pip install tensorflow==1.13.1
pip install humanfriendly Pillow pandas tqdm
python CameraTraps/detection/run_tf_detector_batch.py wherever_you_put_the_detector_file/megadetector_v3.pb some_image_file.jpg some_output_file.json
```


### 3. Batch processing API

Speaking of which, when we process loads of images from collaborators, we use our [batch processing API](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing), which we can make available externally on request.  [Email us](mailto:cameratraps@microsoft.com) for more information.


## Citing MegaDetector

If you use the MegaDetector in a publication, please cite: 
```
@article{beery2019efficient,
  title={Efficient Pipeline for Camera Trap Image Review},
  author={Beery, Sara and Morris, Dan and Yang, Siyu},
  journal={arXiv preprint arXiv:1907.06772},
  year={2019}
}
```

## Tell me more about why detectors are a good first step for camera trap images

Can do!  See these &ldquo;<a href="http://dmorris.net/misc/cameratraps/ai4e_camera_traps_overview">slides</a>&rdquo;.

## Pretty picture

Here&rsquo;s a &ldquo;teaser&rdquo; image of what detector output looks like:

![alt text](images/detector_example.jpg "Red bounding box on fox")

Image credit University of Washington.


## Mesmerizing video

Here&rsquo;s a neat <a href="http://dolphinvm.westus2.cloudapp.azure.com/video/detector_video.html">video</a> of our v2 detector running in a variety of ecosystems, on locations unseen during training.

<a href="http://dolphinvm.westus2.cloudapp.azure.com/video/detector_video.html"><img width=600 src="http://dolphinvm.westus2.cloudapp.azure.com/video/mvideo.jpg"></a><br/>

Image credit <a href="https://emammal.si.edu/">eMammal</a>.


## Can you share the training data?

This model is trained on bounding boxes from a variety of ecosystems, and many of the images we use in training are not publicly-shareable for license reasons.  We do train in part on bounding boxes from two public data sets:

- [Caltech Camera Traps](http://lila.science/datasets/caltech-camera-traps)
- [Snapshot Serengeti](http://lila.science/datasets/snapshot-serengeti)

...so if our detector performs really well on those data sets, that&rsquo;s great, but it&rsquo;s a little bit cheating, because we haven&rsquo;t published the set of locations from those data sets that we use during training.
