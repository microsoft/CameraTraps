# Announcement

At the core of our mission is the desire to create a harmonious space where conservation scientists from all over the globe can unite, share, and grow. We are expanding the CameraTraps repo to introduce **Pytorch-Wildlife**, a Collaborative Deep Learning Framework for Conservation, where researchers can come together to share and use datasets and deep learning architectures for wildlife conservation.
 
We've been inspired by the potential and capabilities of Megadetector, and we deeply value its contributions to the community. **As we forge ahead with Pytorch-Wildlife, under which Megadetector now resides, please know that we remain committed to supporting, maintaining, and developing Megadetector, ensuring its continued relevance, expansion, and utility.**

To use the newest version of MegaDetector with all the exisitng functionatlities, you can use our newly developed [user interface](#explore-pytorch-wildlife-and-megadetector-with-our-user-interface) or simply load the model with **Pytorch-Wildlife** and the weights will be automatically downloaded:

```python
from PytorchWildlife.models import detection as pw_detection
detection_model = pw_detection.MegaDetectorV5()
```

If you'd like to learn more about **Pytorch-Wildlife**, please continue reading.

For those interested in accessing the previous MegaDetector repository, which utilizes the same `MegaDetector v5` model weights and was primarily developed by Dan Morris during his time at Microsoft, please visit the [archive](./archive) directory, or you can visit this [forked repository](https://github.com/agentmorris/MegaDetector/tree/main) that Dan Morris is actively maintaining.
 
**If you have any questions regarding MegaDetector and Pytorch-Wildlife, please <a href="mailto:zhongqimiao@microsoft.com">email us</a>!**

# Table of Contents
- [Announcement](#announcement)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Setup](#setup)
  - [Installation](#installation)
  - [Verifying that CUDA is available (and dealing with the case where it isn't)](#verifying-that-cuda-is-available-and-dealing-with-the-case-where-it-isnt)
  - [Optional steps to make classification faster in Linux](#optional-steps-to-make-classification-faster-in-linux)
  - [Directory Structure](#directory-structure)
  - [Environment Variables](#environment-variables)
  - [Mypy Type Checking](#mypy-type-checking)
- [MegaClassifier](#megaclassifier)
- [Run a trained classifier on new images](#run-a-trained-classifier-on-new-images)
  - [1. Run MegaDetector](#1-run-megadetector)
  - [2. Crop images](#2-crop-images)
  - [3. Run classifier](#3-run-classifier)
  - [4. (Optional) Map MegaClassifier categories to desired categories](#4-optional-map-megaclassifier-categories-to-desired-categories)
  - [5. Merge classification results with detection JSON](#5-merge-classification-results-with-detection-json)
- [Typical training pipeline](#typical-training-pipeline)
  - [1. Select classification labels for training](#1-select-classification-labels-for-training)
  - [2. Query MegaDB for labeled images](#2-query-megadb-for-labeled-images)
  - [3. For images without ground-truth bounding boxes, generate bounding boxes using MegaDetector](#3-for-images-without-ground-truth-bounding-boxes-generate-bounding-boxes-using-megadetector)
  - [4. Create classification dataset and split image crops into train/val/test sets by location](#4-create-classification-dataset-and-split-image-crops-into-trainvaltest-sets-by-location)
  - [5. (Optional) Manually inspect dataset](#5-optional-manually-inspect-dataset)
  - [6. Train classifier](#6-train-classifier)
    - [Note about TensorFlow implementation](#note-about-tensorflow-implementation)
  - [7. Evaluate classifier](#7-evaluate-classifier)
  - [8. Analyze classification results](#8-analyze-classification-results)
  - [9. Export classification results as JSON](#9-export-classification-results-as-json)
  - [10. (Optional) Identify potentially mislabeled images](#10-optional-identify-potentially-mislabeled-images)
- [Miscellaneous Scripts](#miscellaneous-scripts)
- [Label Specification Syntax](#label-specification-syntax)
  - [CSV](#csv)
  - [JSON](#json)


# Overview

This README describes how to train and run an animal "species" classifier. "Species" is in quotes, because the classifier can be trained to identify animals at arbitrary levels within the biological taxonomy of animals.

This guide is written for internal use at Microsoft AI for Earth. Certain services, such as MegaDB and various private repos are only accessible internally within Microsoft. However, this guide may still be of interest to more technical users of the AI for Earth Camera Trap services.

The classifiers trained with this pipeline are intended to be used in conjunction with <a href="https://github.com/ecologize/CameraTraps/blob/master/megadetector.md">MegaDetector</a>, i.e., we use MegaDetector to find animals and crop them out, and we train/run our classifiers on those crops.


# Setup

## Installation

Install Anaconda or [miniconda3](https://docs.conda.io/en/latest/miniconda.html). Then create the conda environment using the following command:

```bash
conda env update -f environment-classifier.yml --prune
```

Activate this conda environment:

```bash
conda activate cameratraps-classifier
```

## Verifying that CUDA is available (and dealing with the case where it isn't)

Verify that CUDA is available (assumes that the current working directory is the CameraTraps repo root):

```bash
python sandbox/torch_test.py
```

If CUDA isn't available but should be (i.e., you have an NVIDIA GPU and recent drivers)...

YMMV, but in at least one Linux environment, the following fixed this issue:

```bash
pip uninstall torch torchvision
conda install pytorch=1.10.1 torchvision=0.11.2 -c pytorch
```

YMMV again, but in at least one Windows environment, the following fixed this issue:

```bash
pip uninstall torch torchvision
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## Optional steps to make classification faster in Linux

If you are on Linux, you may also get some speedup by installing the [accimage](https://github.com/pytorch/accimage) package for acclerated image loading.  Because this is Linux-only and optional, we have commented it out of the environment file, but you can install it with:

```bash
conda install -c conda-forge accimage
```

Similarly, on Linux, you may get some speedup by installing [Pillow-SIMD](https://github.com/uploadcare/pillow-simd):

```bash
pip uninstall -y pillow
pip install pillow-simd
```

## Directory Structure

The classifier pipeline assumes the following directories:

```
classifier-training/            # Azure container mounted locally
    mdcache/                    # cached MegaDetector outputs
        v4.1/
            datasetX.json
    megadb_mislabeled/          # known mislabeled images in MegaDB
        datasetX.csv
    megaclassifier/             # files relevant to MegaClassifier

images/                         # (optional) local directory to save full-size images
    datasetX/                   # images are organized by dataset
        img0.jpg

crops/                          # local directory to save cropped images
    datasetX/                   # images are organized by dataset
        img0___crop00.jpg

CameraTraps/                    # this git repo
    classification/
        BASE_LOGDIR/            # classification dataset and splits
            LOGDIR/             # logs and checkpoints from a single training run

camera-traps-private/           # internal taxonomy git repo
    camera_trap_taxonomy_mapping.csv  # THE taxonomy CSV file
```


## Environment Variables

The following environment variables are useful to have in `.bashrc`:

```bash
# Python development
export PYTHONPATH="/path/to/repos/CameraTraps:/path/to/repos/ai4eutils"
export MYPYPATH=$PYTHONPATH

# accessing MegaDB
export COSMOS_ENDPOINT="[INTERNAL_USE]"
export COSMOS_KEY="[INTERNAL_USE]"

# running Batch API
export BATCH_DETECTION_API_URL="http://[INTERNAL_USE]/v3/camera-trap/detection-batch"
export CLASSIFICATION_BLOB_STORAGE_ACCOUNT="[INTERNAL_USE]"
export CLASSIFICATION_BLOB_CONTAINER="classifier-training"
export CLASSIFICATION_BLOB_CONTAINER_WRITE_SAS="[INTERNAL_USE]"
export DETECTION_API_CALLER="[INTERNAL_USE]"
```


## Mypy Type Checking

Invoke `mypy` from main CameraTraps repo directory. To type check all files in the classifications folder, run

```bash
mypy -p classification
```

To type check a specific script (e.g., `train_classifier.py`) in the classifications folder, run

```bash
mypy -p classification.train_classifier
```


# MegaClassifier

MegaClassifier is an image classifier. MegaClassifier v0.1 is based on an EfficientNet architecture, [implemented in PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch). It supports 169 categories*, where each category is either a single biological taxon or a group of related taxa. See the [`megaclassifier_label_spec.ipynb`](https://github.com/ecologize/CameraTraps/blob/master/classification/megaclassifier_label_spec.ipynb) notebook for more details on the categories. The taxonomy used is based on the 2020_09 revision of the taxonomy CSV.

The training dataset, splits, and parameters used for v0.1 can be found in `classifier-training/megaclassifier/v0.1_training`. There are two variants of MegaClassifier v0.1. Their average top-1 accuracy (recall) and average top-3 accuracy on the test set across all 169 categories are shown in this table:

model name           | architecture    | avg top-1 acc | avg top-3 acc
---------------------|-----------------|---------------|--------------
v0.1_efficientnet-b1 | EfficientNet-B1 | 80.973%       | 91.540%
v0.1_efficientnet-b3 | EfficientNet-B3 | 82.755%       | 92.615%

*Unfortunately, there are some duplicated taxa. Ideally, these should be corrected in the next revision of the taxonomy CSV. The known list of duplicates includes:
* _domestic dogs_: sometimes tagged as species "Canis familiaris" and other times tagged as subspecies "Canis lupus familiaris" (see [Wikipedia](https://en.wikipedia.org/wiki/Dog))
* _zebras_: usually tagged as a species under the genus "equus" but occasionally tagged under the genus "zebra" (see [Wikipedia](https://en.wikipedia.org/wiki/Zebra) and [GBIF](https://www.gbif.org/species/3239462))


# Run a trained classifier on new images

This section explains how to run MegaClassifier on new images. To run MegaClassifier on images already in MegaDB, see the [_Evaluate classifier_](#7-evaluate-classifier) section below.

## 1. Run MegaDetector

First, you need to run MegaDetector on your new images to get an output JSON file, typically using [run_detector_batch.py](https://github.com/ecologize/CameraTraps/blob/main/detection/run_detector_batch.py), though it's also fine to use a third-party tool like [EcoAssist](https://github.com/PetervanLunteren/EcoAssist).  Instructions for running MegaDetector (any version) are [here](https://github.com/ecologize/CameraTraps/blob/main/megadetector.md#using-the-model).


## 2. Crop images

Run `crop_detections.py` to crop the bounding boxes according to the detections JSON. Pass in an Azure Blob Storage container URL if the images are not stored locally. The crops are saved to `/path/to/crops`. Unless you have a good reason not to, use the `--square-crops` flag, which crops the tightest square enclosing each bounding box (which may have an arbitrary aspect ratio).

```bash
python crop_detections.py \
    detections.json \
    /path/to/crops \
    --images-dir /path/to/images \
    --container-url "https://account.blob.core.windows.net/container?sas_token" \
    --threshold 0.8 \
    --save-full-images --square-crops \
    --threads 50 \
    --logdir "."
```

## 3. Run classifier

Load the TorchScript-compiled model file for the classifier. A normal PyTorch checkpoint (e.g., with a `state_dict`) will not work here. For example, MegaClassifier's compiled model file can be found at `classifier-training/megaclassifier/v0.1_efficientnet-b3_compiled.pt`.

The following script will output a CSV file (optionally gzipped) whose columns are:
* `path`: path to image crop, relative to the cropped images directory
* category names: one column per classifier output category. The values are the confidence of the classifier on each category.

On a GPU, this should run at ~200 crops per second.

```bash
python run_classifier.py \
    /path/to/classifier-training/megaclassifier/v0.1_efficientnet-b3_compiled.pt \
    /path/to/crops \
    classifier_output.csv.gz \
    --detections-json detections.json \
    --classifier-categories /path/to/classifier-training/megaclassifier/v0.1_index_to_name.json \
    --image-size 300 --batch-size 64 --num-workers 8
```

## 4. (Optional) Map MegaClassifier categories to desired categories

MegaClassifier outputs 100+ categories, but we usually don't care about all of them. Instead, we can group the classifier labels into desired "target" categories. This process involves 3 sub-steps:

* Specify the target categories that we care about.
* Build a mapping from desired target categories to MegaClassifier labels.
* Aggregate probabilities from the classifier's outputs according to the mapping.

**Specify the target categories that we care about.**

Use the [label specification syntax](#label-specification-syntax) to specify the taxa and/or dataset classes that constitute each target category. If using the CSV format, convert it to the JSON specification syntax using `python csv_to_json.py`.

**Build a mapping from desired target categories to MegaClassifier labels.**

Run the `map_classification_categories.py` script with the target label specification JSON to create a mapping from target categories to MegaClassifier labels. The output file is another JSON file representing a dictionary whose keys are target categories and whose values are lists of MegaClassifier labels. MegaClassifier labels who are not explicitly assigned a target are assigned to a target named "other". Each MegaClassifier label is assigned to exactly one target category.

```bash
python map_classification_categories.py \
    target_label_spec.json \
    /path/to/classifier-training/megaclassifier/v0.1_label_spec.json \
    /path/to/camera-traps-private/camera_trap_taxonomy_mapping.csv \
    --output target_to_classifier_labels.json \
    --classifier-label-index /path/to/classifier-training/megaclassifier/v0.1_index_to_name.json
```

**Aggregate probabilities from the classifier's outputs according to the mapping.**

Using the mapping, create a new version of the classifier output CSV with probabilities summed within each target category. Also output a new "index-to-name" JSON file which identifies the sequential order of the target categories.

```bash
python aggregate_classifier_probs.py \
    classifier_output.csv.gz \
    --target-mapping target_to_classifier_labels.json \
    --output-csv classifier_output_remapped.csv.gz \
    --output-label-index label_index_remapped.json
```

## 5. Merge classification results with detection JSON

Finally, merge the classification results CSV with the original detection JSON file. Use the `--threshold` argument to exclude predicted categories from the JSON file if their confidence is below a certain threshold. The output JSON file path is specified by the `--output-json` argument. If desired, this file can then be opened in Timelapse (requires v2.2.3.7.1 or greater).

```bash
python merge_classification_detection_output.py \
    classifier_output_remapped.csv.gz \
    label_index_remapped.json \
    --output-json detections_with_classifications.json \
    --classifier-name megaclassifier_v0.1_efficientnet-b3 \
    --threshold 0.05 \
    --detection-json detections.json
```


# Typical training pipeline

Before doing any model training, create a directory under `CameraTraps/classification/` for tracking all of our generated files. We refer to this directory with the variable `$BASE_LOGDIR`.

## 1. Select classification labels for training

Create a classification label specification JSON file (usually named `label_spec.json`). This file defines the labels that our classifier will be trained to distinguish, as well as the original dataset labels and/or biological taxa that will map to each classification label. See the required format [here](#json).

For MegaClassifier, see `megaclassifier_label_spec.ipynb` to see how the label specification JSON file is generated.

For bespoke classifiers, it is likely easier to write a CSV file instead of manually writing the JSON file. We then translate to JSON using `csv_to_json.py`. The CSV syntax can be found [here](#csv).


## 2. Query MegaDB for labeled images

In `json_validator.py`, we validate the classification labels specification JSON file. It checks that the specified taxa are included in the master taxonomy CSV file, which specifies the biological taxonomy for every dataset label in MegaDB. The script then queries MegaDB to list all images that match the classification labels specification, and optionally verifies that each image is only assigned a single classification label.

There are some known mislabeled images in MegaDB. These mistakes are currently tracked as CSV files (1 per dataset) in `classifier-training/megadb_mislabeled`. Use the `--mislabeled-images` argument to provide `json_validator.py` the path to these CSVs, so it can ignore or correct the mislabeled images after it queries MegaDB.

The output of `json_validator.py` is another JSON file (`queried_images.json`) that maps image names to a dictionary of properties. The `"label"` key of each entry is a list of strings. For now, this list should only include a single string. However, we are using a list to provide flexibility for allowing multi-label multi-class classification in the future.

```javascript
{
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  // class from dataset in MegaDB
        "bbox": [{"category": "animal",
                  "bbox": [0, 0.347, 0.237, 0.257]}],
        "label": ["cat"]  // labels to use in classifier
    },
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  // class from dataset in MegaDB
        "label": ["cat"]  // labels to use in classifier
    },
    ...
}
```

Example usage of `json_validator.py`:

```bash
python json_validator.py \
    $BASE_LOGDIR/label_spec.json \
    /path/to/camera-traps-private/camera_trap_taxonomy_mapping.csv \
    --output-dir $BASE_LOGDIR \
    --json-indent 1 \
    --mislabeled-images /path/to/classifier-training/megab_mislabeled
```


## 3. For images without ground-truth bounding boxes, generate bounding boxes using MegaDetector

While some labeled images in MegaDB already have ground-truth bounding boxes, other images do not. For the labeled images without bounding box annotations, we run MegaDetector to get bounding boxes. MegaDetector can be run either locally or via the Batch Detection API.

This step consists of 3 sub-steps:
1. Run MegaDetector (either locally or via Batch API) on the queried images.
2. Cache MegaDetector results on the images to JSON files in `classifier-training/mdcache`.
3. Download and crop the images to be used for training the classifier.

<details>
    <summary>To run MegaDetector locally</summary>

This option is only recommended if you meet all of the following criteria:
- there are not too many images (<1 million)
- all of your image files are from a single dataset in a single Azure container
- none of the images already have cached MegaDetector results

Download all of the images to `/path/to/images/name_of_dataset`. Then, follow the instructions from the [MegaDetector README](https://github.com/ecologize/CameraTraps/blob/master/megadetector.md) to run MegaDetector. Finally, cache the detection results. The commands should be roughly as follows, assuming your terminal is in the `CameraTraps/` folder:

```bash
# Download the MegaDetector model file
wget -O md_v4.1.0.pb https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# install TensorFlow v1 and other dependences
conda env update -f environment-detector.yml --prune
conda activate cameratraps-detector

# run MegaDetector
python detection/run_tf_detector_batch.py \
    md_v4.1.0.pb \
    /path/to/images/name_of_dataset \
    classification/$BASE_LOGDIR/detections.json \
    --recursive --output_relative_filenames

# cache the detections
python cache_batchapi_outputs.py \
    classification/$BASE_LOGDIR/detections.json \
    --format detections \
    --dataset name_of_dataset \
    --detector-output-cache-dir "/path/to/classifier-training/mdcache" --detector-version "4.1"
```
</details>

<details>
    <summary>To use the MegaDetector Batch Detection API</summary>

We use the `detect_and_crop.py` script. In theory, we can do everything we need in a single invocation. The script groups the queried images by dataset and then submits 1 "task" to the Batch Detection API for each dataset. It knows to wait for each task to finish running, before starting to download and crop the images based on bounding boxes. Because of the resume file, in theory it should be OK to cancel the script once the tasks are all submitted, then re-run the script later with the exact same arguments to fetch the results and begin the downloading and cropping.

```bash
python detect_and_crop.py \
    $BASE_LOGDIR/queried_images.json \
    $BASE_LOGDIR \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version 4.1 \
    --run-detector --resume-file $BASE_LOGDIR/resume.json \
    --cropped-images-dir /path/to/crops --square-crops --threshold 0.9 \
    --save-full-images --images-dir /path/to/images --threads 50
```

However, because the Batch Detection API sometimes returns incorrect responses, in practice we may need to call `detect_and_crop.py` multiple times. It is important to understand the 2 different "modes" of the script.

1. Call the Batch Detection API, and cache the results.
    * To run this mode: set the `--run-detector` flag
    * To skip this mode: omit the flag
2. Using ground truth and cached detections, crop the images.
    * To run this mode: set `--cropped-images-dir /path/to/crops`
    * To skip this mode: omit `--cropped-images-dir`

Thus, we will first call the Batch Detection API. This saves a `resume.json` file that contains all of the task IDs. Because the Batch Detection API does not always respond with the correct task status, the only real way to verify if a task has finished running is to check the `async-api-*` Azure Storage container and see if the 3 output files are there.

```bash
python detect_and_crop.py \
    $BASE_LOGDIR/queried_images.json \
    $BASE_LOGDIR \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version 4.1 \
    --run-detector --resume-file $BASE_LOGDIR/resume.json
```

When a task finishes running, manually create a JSON file for each task according to the [Batch Detection API response format](https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#api-outputs). Save the JSON file to `$BASE_LOGDIR/batchapi_response/dataset.json`. Then, use `cache_batchapi_outputs.py` to cache these results:

```bash
python cache_batchapi_outputs.py \
    $BASE_LOGDIR/batchapi_response/dataset.json \
    --dataset dataset \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version 4.1
```
</details>

Once we have detection results (whether from running MegaDetector locally or via the Batch Detection API), the following command crops the images based on the ground truth and detected bounding boxes.
- If the full images are already locally available (e.g., if MegaDetector was run locally), provide their location via `--images-dir` argument.
- If the full images are not locally available (e.g., if using the Batch Detection API), the script will download the images from Azure Blob Storage before cropping them. In this case, if the `--save-full-images` flag is set, the full images are saved to the `--images-dir` folder. If `--save-full-images` is not set, then the full images are discarded after cropping.

On a VM, expect this download and cropping step to run at ~60 images per second (~5 hours for 1 million images). Unless you have a good reason not to, use the `--square-crops` flag, which crops the tightest square enclosing each bounding box (which may have an arbitrary aspect ratio). Because images are resized to a square during training, using square crops guarantees that the model does not see a distorted aspect ratio of the animal.

```bash
python detect_and_crop.py \
    $BASE_LOGDIR/queried_images.json \
    $BASE_LOGDIR \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version 4.1 \
    --cropped-images-dir /path/to/crops --square-crops --threshold 0.9 \
    --save-full-images --images-dir /path/to/images --threads 50
```

## 4. Create classification dataset and split image crops into train/val/test sets by location

Preparing a classification dataset for training involves two steps.

1. Create a CSV file (`classification_ds.csv`) representing our classification dataset, where each row in this CSV represents a single training example, which is an image crop with its label. Along with this CSV file, we also create a `label_index.json` JSON file which defines a integer ordering over the string classification label names.
2. Split the training examples into 3 sets (train, val, and test) based on the geographic location where the images were taken. The split is specified by a JSON file (`splits.json`).

Both of these steps are handled by `create_classification_dataset.py`, and each step is explained in more detail below.

**Creating the classification dataset CSV**

The classification dataset CSV has the columns listed below. Only image crops in `/path/to/crops` from images listed in the `queried_images.json` file are included in the classification dataset CSV. For now, the 'label' column is a single value. However, we support a comma-separated list of labels to provide flexibility for allowing multi-label multi-class classification in the future.

* 'path': str, path to image crop
* 'dataset': str, name of dataset that image is from
* 'location': str, location that image was taken, as saved in MegaDB, or `"unknown_location"` if no location is provided in MegaDB
* 'dataset_class': str, original class assigned to image, as saved in MegaDB
* 'confidence': float, confidence that this crop is of an actual animal, 1.0 if the crop is a "ground truth bounding box" (i.e., from MegaDB), <= 1.0 if the bounding box was detected by MegaDetector
* 'label': str, comma-separated list of label(s) assigned to this crop for classification

The command to create the CSV is shown below. Three arguments merit explanation:

* The `--threshold` argument filters out crops whose detection confidence is below a given threshold. Note, however, that if during the cropping step you only cropped bounding boxes above a detection confidence of 0.9, specifying a threshold of 0.8 here will have the same effect as specifying a threshold of 0.9. This script will not magically go back and crop the bounding boxes with a detection confidence between 0.8 and 0.9.
* The `--min-locs` argument filters out crops whose label appears in fewer than some number of locations. This is useful for targeting a minimum diversity of locations. Because we split images into train/val/test based on location, at the bare minimum you should consider setting `--min-locs 3`. Otherwise, the label will be entirely excluded from at least one of the 3 splits.
* The `--match-test` argument is useful for trying a new training dataset, but using an existing test set. This argument takes two values: `--match-test CLASSIFICATION_CSV SPLITS_JSON`. After creating the classification dataset (ignoring this argument), the script will append all crops from the given `CLASSIFICATION_CSV` whose "location" appears in the test set from `SPLITS_JSON`.

The script also creates a `label_index.json` file, which maps integers to label names. The keys are string representations of Python integers (JSON requires keys to be strings), numbered from 0 to `num_labels-1`. The values are the label names (strings).

```bash
python create_classification_dataset.py \
    $BASE_LOGDIR \
    --mode csv \
    --queried-images-json $BASE_LOGDIR/queried_images.json \
    --cropped-images-dir /path/to/crops \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version 4.1 \
    --threshold 0.9 \
    --min-locs 20
```

**Splitting training examples by geographic location**

We split training examples by geographic location in order to test the classifier's ability to generalize to unseen locations. Otherwise, the classifier might "memorize" the distribution of known species at each location. This second step assumes that a classification dataset CSV (`classification_ds.csv`) already exists and creates a `splits.json` file. This `splits.json` file maps each split `['train', 'val', 'test']` to a list of length-2 lists, where each inner list is `[<dataset>, <location>]`. In other words, each "geographic location" is identified by a (dataset, dataset-location) pair.

Several arguments merit explanation:

* `--val-frac` and `--test-frac`: These arguments specify roughly the fraction of all crops that should be put into the val and test sets, respectively. How this is done depends on the `--method` argument, explained below. Note that `--match-test` and `--test-frac` are mutually exclusive arguments, and `--test-frac` is treated as 0 if it is not given. The size of the train set is `1 - val_frac - test_frac`.
* `--method`: There are two heuristics to choose from which determine how the splits are made:
  * The `random` heuristic tries 10,000 different combinations of assigning approximately `--val-frac` and `--test-frac` locations to the val and test sets, with the remaining going to the train set. It then scores each of these combinations on how far each label's distribution of image crops and locations are from the desired train/val/test split. The combination with the lowest score (lower is better) is selected.
  * The `smallest_first` heuristic sorts the labels from fewest to most examples, and then sorts the locations for each label from fewest to most examples. Locations are added in order to the test, val, then train sets until each split meets the desired split size.
* `--label-spec`: This argument is useful for prioritizing certain datasets over others for inclusion in the test set, based on the given label specification JSON file. This argument requires `--method=smallest_first`.
* The `--match-test` argument is useful for trying a new training dataset, but using an existing test set. This argument takes two values: `--match-test CLASSIFICATION_CSV SPLITS_JSON`. This will simply copy the test set from `SPLITS_JSON`.

```bash
python create_classification_dataset.py \
    $BASE_LOGDIR \
    --mode splits \
    --val-frac 0.2 --test-frac 0.2 \
    --method random
```

**Combining both steps in one command**

```bash
python create_classification_dataset.py \
    $BASE_LOGDIR \
    --mode csv splits \
    --queried-images-json $BASE_LOGDIR/queried_images.json \
    --cropped-images-dir /path/to/crops \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version 4.1 \
    --threshold 0.9 \
    --min-locs 20 \
    --val-frac 0.2 --test-frac 0.2 \
    --method random
```

## 5. (Optional) Manually inspect dataset

Copy the `inspect_dataset.ipynb` notebook into `$BASE_LOGDIR`. Open a Jupyter lab or notebook instance, and run the notebook. This notebook:

* Samples some images from each label and displays them. Verify that these images are reasonable.
* Prints out the distribution of images and locations across the train, val, and test splits, and highlights the labels that have "extreme" distributions. Verify that these deviations are acceptable.

## 6. Train classifier

Use the `train_classifier.py` script to train a classifier in PyTorch based on either a Resnet ([torchvision](https://pytorch.org/docs/stable/torchvision/models.html#id10)) or EfficientNet ([lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)) architecture. The script runs the model on the val split after every training epoch, and if the val split achieves a new highest accuracy, it saves the model weights from that epoch and also runs the model on the test split. If the model does not improve the val accuracy for 8 epochs, the model stops training. Finally, the best-performing model (based on val accuracy) is evaluated on all three splits (see the next section).

The script assumes that the `$BASE_LOGDIR` directory has 3 files: 1) classification dataset CSV, 2) label index JSON, and 3) splits JSON.

Most command-line options are self-explanatory. However, several merit explanation:

* `--pretrained`: Without this argument, the model is trained from scratch. If this argument is used as a flag (without a value), then the model uses pre-trained weights from an ImageNet-trained model. This argument can also take a path to a normal model checkpoint (not a TorchScript-compiled checkpoint) as well. This is useful, for example, to use MegaClassifier's weights as the starting point for a bespoke classifier.
* `--label-weighted`: Instead of a simple random shuffle, this flag causes training examples to be selected through a weighted sampling procedure. Examples are weighted inversely proportional to number of examples in each label. In other words, all labels get sampled with equal frequency. This effectively balances the dataset. We found that weighted sampling was more effective than weighting each example's loss because the weights varied dramatically between labels (e.g., often exceeding 100x between the smallest and largest labels). If using a weighted loss, certain batches would have an extremely large loss and gradient, which was detrimental to training.
* `--weight-by-detection-conf`: If used as a flag, this argument weights each example by its detection confidence. This argument optionally takes a path to a compressed numpy archive (`.npz`) file containing the isotonic regression interpolation coordinates for calibrating the detection confidence.
* `--log-extreme-examples`: This flag specifies the number of true-positive (tp), false-positive (fp), and false-negative (fn) examples of each label to log in TensorBoard during each epoch of training. This flag is very helpful for identifying what images the classifier is struggling with during training. However, it is recommended to turn this flag OFF when training MegaClassifier because its RAM usage is linearly proportional to the number of classes (and MegaClassifier has a lot of classes).
* `--finetune`: If used as a flag, this argument will only adjust the final fully-connected layer of the model. This argument optionally takes an integer, which specifies the number of epochs for fine-tuning the final layer before enabling all layers to be trained. I found that empirically there was no observable benefit to fine-tuning the final layer first before training all layers, so usually there should be no reason to use this argument.

```bash
python train_classifier.py \
    $BASE_LOGDIR \
    /path/to/crops \
    --model-name efficientnet-b3 --pretrained \
    --label-weighted --weight-by-detection-conf /path/to/classifier-training/mdv4_1_isotonic_calibration.npz \
    --epochs 50 --batch-size 160 --lr 0.0001 \
    --logdir $BASE_LOGDIR --log-extreme-examples 3
```

The following hyperparameters for MegaClassifier seem to work well for both EfficientNet-B1 and EfficientNet-B3 (PyTorch implementation):

* no initial finetuning
* `--pretrained`
* EfficientNet-B1: `--batch-size 192` (on 2 GPUs), EfficientNet-B3: `--batch-size 160` (on 4 GPUs)
* `--label-weighted`
* `--epochs 50`: test-set accuracy will likely plateau before the full 50 epochs
* `--weight-by-detection-conf /path/to/mdv4_1_isotonic_calibration.npz`
* `--lr 3e-5`
* `--weight-decay 1e-6`: values between `1e-5` and `1e-6` are generally OK

During training, logs are written to TensorBoard at the end of each epoch. View the logs by running the following command. Setting `--samples_per_plugin` to a large number (e.g., 10,000) prevents TensorBoard from omitting images. However, this may incur high RAM usage.

```bash
tensorboard --logdir $BASE_LOGDIR --bind_all --samples_per_plugin images=10000
```

### Note about TensorFlow implementation

There is a `train_classifier_tf.py` script in this directory which attempts to mimic the PyTorch training script, but using TensorFlow v2 instead. The reason I tried to do a TensorFlow implementation is because TensorFlow v2.3 introduced an official Keras EfficientNet implementation, whereas the PyTorch EfficientNet was a third-party implementation. However, the TensorFlow implementation proved difficult to implement well, and several features from the PyTorch version are different or remain lacking in the TensorFlow version:

* Training on multiple GPUs is not supported. PyTorch natively supports this by wrapping a model in `torch.nn.DataParallel`. TensorFlow also supports this when using `tf.keras.Model.fit()`. However, I wrote my own training loop in TensorFlow to match the PyTorch version instead of using `model.fit()`. Adopting `model.fit()` would likely require a completely different implementation from the PyTorch code and involve subclassing `tf.keras.Model` to define a new `model.train_step()` method. For now, the TensorFlow code is limited to a single GPU, which means using extremely small batch sizes.
* `--label-weighted` uses weighted loss instead of weighted data sampling. Weighted data sampling can be implemented in TensorFlow, but it is much more verbose than the equivalent PyTorch code. See the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#oversampling) on oversampling the minority class.
* initializing the model from a checkpoint, e.g., `--pretrained /path/to/checkpoint`

Consequently, I recommend against using the TensorFlow training code until these issues are resolved.


## 7. Evaluate classifier

After training a model, we evaluate its performance using `evaluate_model.py`. Technically, this step is actually already run by the model training code (`train_classifier.py`), so usually you do not need to run `evaluate_model.py` on your own. However, there are two situations where you would want to run this step manually:

* if the training code somehow runs into an error during model evaluation, or
* if you want to evaluate a model (e.g., MegaClassifier) on a different dataset than the model was trained on. See section below on comparing the performance of MegaClassifier vs. a bespoke classifier.

During training, we already output some basic overall statistics about model performance, but during evaluation, we generate a more picture of model performance, captured in the following output files:

1. `outputs_{split}.csv`: one file per split, contains columns:
    * `'path'`: str, path to cropped image
    * `'label'`: str
    * `'weight'`: float
    * `[label names]`: float, confidence in each label

2. `overall_metrics.csv`, contains columns:
    * `'split'`: str
    * `'loss'`: float, mean per-example loss over entire epoch
    * `'acc_top{k}'`: float, accuracy@k over the entire epoch
    * `'loss_weighted'` and `'acc_weighted_top{k}'`: float, weighted versions

3. `confusion_matrices.npz`: keys are splits (`['train', 'val', 'test']`), values are `np.ndarray` confusion matrices

4. `label_stats.csv`: per-label statistics, columns are `['split', 'label', 'precision', 'recall']`

The `evaluate_model.py` script takes 2 main inputs: a `params.json` file created during model training, and a path to a model checkpoint.

* `params.json`: Passing in a `params.json` file simplifies the number of arguments you need to pass to `evaluate_model.py`, although there are many parameters which can be overridden via command-line arguments (run `evaluate_model.py --help` to see which parameters can be overridden).
* checkpoint file: This can be either a normal checkpoint or a TorchScript-compiled checkpoint. If the checkpoint is a normal checkpoint, the script will compile a TorchScript checkpoint and save it to the same place as the normal checkpoint, except with a `_compiled` suffix added to the filename.

Note that the classifier evaluation code uses the "accimage" backend for image transformations instead of the "Pillow" or "Pillow-SIMD" backend used during training. The accimage backend empirically improves data loading speed by about 20-50% over Pillow and Pillow-SIMD. However, accimage runs into occasional unpredictable errors every once in a while, so it is impractical for training. For evaluation, it has worked quite well though.

There tends to be a small difference in the val and test accuracies between training and evaluation. This might be due to the differences between Pillow and accimage, although I haven't rigorously tested it to be sure.

```bash
python evaluate_model.py \
    $BASE_LOGDIR/$LOGDIR/params.json \
    $BASE_LOGDIR/$LOGDIR/ckpt_XX.pt \
    --output-dir $BASE_LOGDIR/$LOGDIR \
    --splits train val test \
    --batch-size 256
```

**Compare MegaClassifier performance vs. bespoke classifier**

We can also use compare the performance of MegaClassifier vs. a bespoke classifier. First, generate a mapping from MegaClassifier categories to the bespoke classifier's categories using `map_classification_categories.py`. If the bespoke classifier's label specification JSON file was defined using taxa, then you may be able to directly use that JSON file. However, if the label specification was defined using dataset-specific classes, you should consider writing a broader label specification using taxa. See the instructions from [_Map MegaClassifier categories to desired categories_](#4-optional-map-megaclassifier-categories-to-desired-categories).

Next, run `evaluate_mode.py`, passing in the MegaClassifier compiled checkpoint as well as the new category mapping and MegaClassifier's own label index JSON file.

The following script shows example commands. It assumes that the bespoke classifier's log directory is `$BASE_LOGDIR`.

```bash
python map_classification_categories.py \
    $BASE_LOGDIR/label_spec.json \
    /path/to/classifier-training/megaclassifier/v0.1_label_spec.json \
    /path/to/camera-traps-private/camera_trap_taxonomy_mapping.csv \
    -o run_idfg4/megaclassifier/target_to_megaclassifier_labels.json \
    --classifier-label-index /path/to/classifier-training/megaclassifier/v0.1_index_to_name.json

 python evaluate_model.py \
    $BASE_LOGDIR/$LOGDIR/params.json \
    /path/to/classifier-training/megaclassifier/v0.1_efficientnet-b3_compiled.pt \
    --output-dir $BASE_LOGDIR/megaclassifier \
    --splits test \
    --target-mapping $BASE_LOGDIR/megaclassifier/target_to_megaclassifier_labels.json \
    --label-index /path/to/classifier-training/megaclassifier/v0.1_index_to_name.json \
    --model-name efficientnet-b3 --batch-size 256 --num-workers 12
```


## 8. Analyze classification results

Copy the `analyze_classifier_results.ipynb` notebook into `$BASE_LOGDIR/$LOGDIR`. Open a Jupyter lab or notebook instance, and run the notebook. This notebook:

* Plots the confusion matrix for each split.
* Prints out the precision / recall values for each label.
* Plots classifier's calibration curve for each label.
* For a given label, plots images from that label, grouped by their predicted class.


## 9. Export classification results as JSON

Once we have the `output_{split}.csv.gz` files, we can export our classification results in the Batch Detection API JSON format. The following command generates such a JSON file for the images from the test set, including only classification probabilities greater than 0.1, and also including the true label:

```bash
python merge_classification_detection_output.py \
    $BASE_LOGDIR/$LOGDIR/outputs_test.csv.gz \
    $BASE_LOGDIR/label_index.json \
    --output-json $BASE_LOGDIR/$LOGDIR/outputs_test.json \
    --classifier-name "myclassifier" \
    --threshold 0.1 \
    --queried-images-json $BASE_LOGDIR/queried_images.json \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version "4.1" \
    --label last
```

## 10. (Optional) Identify potentially mislabeled images

We can now use our trained classifier to identify potentially mislabeled images by looking at the model's false positives. A "mislabeled candidate" is defined as an image meeting both of the following criteria:
- according to the ground-truth label, the model made an incorrect prediction
- the model's prediction confidence exceeds its confidence for the ground-truth label by some minimum confidence.

At this point, we should have the following folder structure:
```
BASE_LOGDIR/
    queried_images.json           # generated in step (2)
    label_index.json              # generated in step (4)
    LOGDIR/                       # generated in step (6)
        outputs_{split}.csv.json  # generated in step (7)
```

We generate a JSON file that can be loaded into Timelapse to help us review mislabeled candidates. We again use `merge_classification_detection_output.py`. However, instead of outputting raw classification probabilities, we output the margin of error by passing the `--relative-conf` flag.

```bash
python merge_classification_detection_output.py \
    $BASE_LOGDIR/$LOGDIR/outputs_test.csv.gz \
    $BASE_LOGDIR/label_index.json \
    --output-json $BASE_LOGDIR/$LOGDIR/outputs_json_test_set_relative_conf.json \
    --classifier-name "myclassifier" \
    --queried-images-json $BASE_LOGDIR/queried_images.json \
    --detector-output-cache-dir /path/to/classifier-training/mdcache --detector-version "4.1" \
    --relative-conf
```

If the images are not already on the Timelapse machine, and we don't want to download the entire dataset onto the Timelapse machine, we can instead choose to only download the mislabeled candidate images. We use the `identify_mislabeled_candidates.py` script to generate the lists of images to download, one file per split and dataset: `$LOGDIR/mislabeled_candidates_{split}_{dataset}.txt`. It is recommended to set a high margin >=0.95 in order to restrict ourselves to only the most-likely mislabeled candidates. Then, use either AzCopy or `data_management/megadb/download_images.py` to do the actual downloading.

Using `data_management/megadb/download_images.py` is the recommended and faster way of downloading images. It expects a file list with the format `<dataset_name>/<blob_name>`, so we have to pass the `--include-dataset-in-filename` flag to `identify_mislabeled_candidates.py`.

```bash
python identify_mislabeled_candidates.py $BASE_LOGDIR/$LOGDIR \
    --margin 0.95 --splits test --include-dataset-in-filename

python ../data_management/megadb/download_images.py txt \
    $BASE_LOGDIR/$LOGDIR/mislabeled_candidates_{split}_{dataset}.json \
    /save/images/to/here \
    --threads 50
```

Until AzCopy improves its performance for its undocumented `--list-of-files` option, [its performance is generally much slower](https://github.com/Azure/azure-storage-azcopy/issues/1152). However, we can use it as follows:

```bash
python identify_mislabeled_candidates.py $BASE_LOGDIR/$LOGDIR \
    --margin 0.95 --splits test

azcopy cp "http://<url_of_container>?<sas_token>" "/save/files/here" \
    --list-of-files "mislabeled_candidates_{split}_{dataset}.txt"
```

Load the images into Timelapse with a template that includes a Flag named "mislabeled" and a Note named "correct_class". Load the JSON classifications file, and enable the image recognition controls. There are two methods for effectively identifying potential false positives. Whenever you identify a mislabeled image, check the "mislabeled" checkbox. If you know its correct class, type it into the "correct_class" text field.

1. If you downloaded images using `identify_mislabeled_candidates.py`, then select images with "label: elk", for example. This should show all images that are labeled "elk" but predicted as a different class with a margin of error of at least 0.95. Look through the selected images, and any image that is *not* actually of an elk is therefore mislabeled.

2. If you already had all the images downloaded, then select images with "elk", but set the confidence threshold to >=0.95. This will show all images that the classifier incorrectly predicted as "elk" by a margin of error of at least 0.95. Look through the selected images, and any image that *is* actually an elk is therefore mislabeled.

When you are done identifying mislabeled images, export the Timelapse database to a CSV file `mislabeled_images.csv`. We can now update our list of known mislabeled images with this CSV:

```bash
python save_mislabeled.py /path/to/classifier-training /path/to/mislabeled_images.csv
```


# Miscellaneous Scripts

* `analyze_failed_images.py`: many scripts in the training pipeline produce log files which list images that either failed during detection, failed to download, or failed to crop. This script analyzes the log files to separate out the images into 5 categories:
  * `'good'`: no obvious issues
  * `'nonexistant'`: image file does not exist in Azure Blob Storage
  * `'non_image'`: file is not a recognized image file (based on file extension)
  * `'truncated'`: file is truncated but can only be opened by Pillow by setting `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True`
  * `'bad'`: file exists, but cannot be opened even when setting `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True`
* `json_to_azcopy_list.py`: Given JSON file such as the `queried_images.json` file output from `json_validator.py`, generates one text file `{dataset}_images.txt` for every dataset included. The text file can then be passed to `azcopy` using the [undocumented](https://github.com/Azure/azure-storage-azcopy/wiki/Listing-specific-files-to-transfer) `--list-of-files` argument to be downloaded. However, until AzCopy fixes its [performance issues](https://github.com/Azure/azure-storage-azcopy/issues/1152) with the `--list-of-files` argument, this is not a recommended method for downloading image files from Azure. Instead, consider using the `data_management/megadb/download_images.py` script instead.


# Label Specification Syntax

## CSV

```
output_label,type,content

# select a specific row from the master taxonomy CSV
<label>,row,<dataset_name>|<dataset_label>

# select all animals in a taxon from a particular dataset
<label>,datasettaxon,<dataset_name>|<taxon_level>|<taxon_name>

# select all animals in a taxon across all datasets
<label>,<taxon_level>,<taxon_name>

# exclude certain rows or taxa
!<label>,...

# set a limit on the number of images to sample for this class
<label>,max_count,<int>

# when sampling images, prioritize certain datasets over others
# is they Python syntax for List[List[str]], i.e., a list of lists of strings
<label>,prioritize,"[['<dataset_name1>', '<dataset_name2>'], ['<dataset_name3>']]"
```

A CSV label specification file can be converted to the [JSON label specification syntax](#json) via the Python script `csv_to_json.py`.


## JSON

```javascript
{
    // name of classification label
    "cervid": {

        // select animals to include based on hierarchical taxonomy,
        // optionally restricting to a subset of datasets
        "taxa": [
            {
                "level": "family",
                "name": "cervidae",
                "datasets": ["idfg", "idfg_swwlf_2019"]
                // include all datasets if no "datasets" key given
            }
        ],

        // select animals to include based on dataset labels
        "dataset_labels": {
            "idfg": ["deer", "elk", "prong"],
            "idfg_swwlf_2019": ["elk", "muledeer", "whitetaileddeer"]
        },

        "max_count": 50000,  // only include up to this many images (not crops)

        // prioritize images from certain datasets over others,
        // only used if "max_count" is given
        "prioritize": [
            ["idfg_swwlf_2019"],  // give 1st priority to images from this list of datasets
            ["idfg"]  // give 2nd priority to images from this list of datasets
            // give remaining priority to images from all other datasets
        ]

    },

    // name of another classification label
    "bird": {
        "taxa": [
            {
                "level": "class",
                "name": "aves"
            }
        ],
        "dataset_labels": {
            "idfg_swwlf_2019": ["bird"]
        },

        // exclude animals using the same format
        "exclude": {
            // same format as "taxa" above
            "taxa": [
                {
                    "level": "genus",
                    "name": "meleagris"
                }
            ],

            // same format as "dataset_labels" above
            "dataset_labels": {
                "idfg_swwlf_2019": ["turkey"]
            }
        }
    }
}
```
