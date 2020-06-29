# Overview

This repo contains the tools for training, running, and evaluating detectors and classifiers for images collected from motion-triggered camera traps.  The core functionality provided is:

- Data parsing from frequently-used camera trap metadata formats into a common format
- Training and evaluation of detectors, particularly our "MegaDetector", which does a pretty good job finding terrestrial animals in a variety of ecosystems
- Training and evaluation of species-level classifiers for specific data sets
- A Web-based demo that runs our models via a REST API that hosts them on a Web endpoint
- Miscellaneous useful tools for manipulating camera trap data
- Research experiments we're doing around camera trap data (i.e., some directories are highly experimental and you should take them with a grain of salt)

Classifiers and detectors are trained using TensorFlow.

This repo is maintained by folks in the [Microsoft AI for Earth](http://aka.ms/aiforearth) program who like looking at pictures of animals.  I mean, we want to use machine learning to support conservation too, but we also really like looking at pictures of animals.

# How we work with ecologists

We work with ecologists all over the world to help them spend less time annotating images and more time thinking about conservation.  You can read a little more about how this works on our [AI for Earth camera trap collaborations page](collaborations.md).

You can also read about what we do to support camera trap researchers in our recent [blog post](https://medium.com/microsoftazure/accelerating-biodiversity-surveys-with-azure-machine-learning-9be53f41e674).


# Data

This repo does not directly host camera trap data, but we work with our collaborators to make data and annotations available whenever possible on [lila.science](http://lila.science).


# Models

This repo does not extensively host species classification models, though we will release models when they are at a level of generality that they might be useful to other people.  But...


## MegaDetector

Speaking of models that might be useful to other people, we have trained a one-class animal detector trained on several hundred thousand bounding boxes from a variety of ecosystems.  Lots more information &ndash; including download links &ndash; on the [MegaDetector page](megadetector.md).

Here's a "teaser" image of what detector output looks like:

![alt text](images/detector_example.jpg "Red bounding box on fox")

Image credit University of Washington.


# Contact

For questions about this repo, contact [cameratraps@microsoft.com](mailto:cameratraps@microsoft.com).


# Contents

This repo is organized into the following folders...


## api

Code for hosting our models as an API, either for synchronous operation (e.g. for real-time inference or for our Web-based demo) or as a batch process (for large biodiversity surveys).


## classification

Code for training species classifiers on new data sets, generally trained on crops generated via an existing detector.  We'll release some classifiers soon, but more importantly, here's a [tutorial](https://github.com/microsoft/CameraTraps/blob/master/classification/TUTORIAL.md) on training your own classifier using our detector and our training pipeline.

Oh, and here's another "teaser image" of what you get at the end of training a classifier:

<img src="images/warthog_classifications.jpg" width="700">

## data_management

Code for:

- Converting frequently-used metadata formats to [COCO Camera Traps](https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format) format
- Creating, visualizing, and  editing COCO Camera Traps .json databases
- Generating tfrecords

## demo

Source for the Web-based demo of our MegaDetector model (we'll release the demo soon!).


## detection

Code for training and evaluating detectors.


## research

Ongoing research projects that use this repository in one way or another; as of the time I'm editing this README, there are projects in this folder around active learning and the use of simulated environments for training data augmentation.


## sandbox

Random things that don't fit in any other directory.  Currently contains a single file, a not-super-useful but super-duper-satisfying and mostly-successful attempt to use OCR to pull metadata out of image pixels in a fairly generic way, to handle those pesky cases when image metadata is lost.


# Installation

We use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage our Python package dependencies. Conda is a package and environment management system. You can install a lightweight distribution of conda (Miniconda) for your OS via installers at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

## Initial setup

### Utility and visualization scripts

The required Python packages for running utility and visualization scripts in this repo are listed in [environment.yml](environment.yml).  To set up your environment for these scripts, in your shell, navigate to the root directory of this repo and issue the following command to create a virtual environment via conda called `cameratraps` (specified in the environment file) and install the required packages:

```
conda env create --file environment.yml
```

For unix users, you need to have gcc installed in order to compile the pip packages. If you do not already have gcc installed, run the following command before creating the conda environment:

```bash
sudo apt update
sudo apt install build-essential
```

### Machine learning scripts

Scripts that execute machine learning code &ndash; specifically, scripts in the folders `api`, `detection`, and `classification` &ndash; require additional depdendencies.  In particular, the `detection/run_tf_detector*.py` scripts should use [environment-detector.yml](environment-detector.yml) to set up the environment, as follows:

```
conda env create --file environment-detector.yml
```

This environment file allows any TensorFlow version from 1.9 to 1.15 to be installed, but you may need to adjust that version for your environment.  Specifically, if you are running on an Azure Data Science Virtual Machine (which has CUDA 10.1 as of the time I'm writing this), you may receive a CUDA error, in which case you should change the line:

`- tensorflow-gpu>=1.9.0, <1.15.0`

...to:

`- tensorflow-gpu=1.13.1`

...before creating your environment.

### Troubleshooting

If you run into an error while creating either of the above environments, try updating conda to version 4.5.11 or above. Check the version of conda using `conda --version`.

## Usage

To enter the conda virtual environment at your current shell, run:

`conda activate cameratraps`

...or, if you used the environment-detector.yml file above:

`conda activate cameratraps-detector`

You should see `(cameratraps)` prepended to the command line prompt. Invoking `python` or `jupyter notebook` will now be using the interpreter and packages available in this virtual env.

To exit the virtual env, issue `conda deactivate`.

## Add additional packages

If you need to use additional packages, add them to the environment file and run

```bash
conda env update --name cameratraps --file environment.yml --prune
```
or
```bash
conda env update --name cameratraps-detector --file environment-detector.yml --prune
```

## Other notes

In some scripts, we also assume that you have the [AI for Earth utilities repo](https://github.com/Microsoft/ai4eutils) (`ai4eutils`) cloned and its path appended to `PYTHONPATH`. You can append a path to `PYTHONPATH` for the current shell session by executing the following on Windows:

```set PYTHONPATH="%PYTHONPATH%;c:\wherever_you_put_the_ai4eutils_repo"```

You can do this with the following on Linux:

```export PYTHONPATH="$PYTHONPATH:/absolute/path/to/repo/ai4eutils"```

Adding this line to your `~/.bashrc` (on Linux) modifies `PYTHONPATH` permanently.

We also do our best to follow [Google's Python Style Guide](http://google.github.io/styleguide/pyguide.html), and we have adopted their `pylintrc` file, with the following differences:
- indent code blocks with 4 spaces (instead of 2)


# Gratuitous pretty camera trap picture

![alt text](images/nacti.jpg "Bird flying above water")

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

## License

This repository is licensed with the [MIT license](https://github.com/Microsoft/dotnet/blob/master/LICENSE).
