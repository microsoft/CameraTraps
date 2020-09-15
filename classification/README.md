# Table of Contents
* [Overview](#overview)
* [Setup](#setup)
  * [Installation](#installation)
  * [Directory Structure](#directory-structure)
  * [Environment Variables](#environment-variables)
* [Running a classifier on new images](#running-a-classifier-on-new-images)
  1. [Run MegaDetector](#1-run-megadetector)
  2. [Crop images](#2-crop-images)
  3. [Run classifier](#3-run-classifier)
  4. [(Optional) Build mapping from desired categories to MegaClassifier categories](#4-optional-build-mapping-from-desired-categories-to-megaclassifier-categories)
  5. [Merge classification results with detection JSON](#5-merge-classification-results-with-detection-json)
* [Typical Training Pipeline](#typical-training-pipeline)
  1. Select classification labels for training.
  2. Validate the classification labels specification JSON file, and generate a list of images to run detection on.
  3. Submit images without ground-truth bounding boxes to the MegaDetector Batch Detection API to get bounding box labels.
  4. Create classification dataset and split into train/val/test sets by location.
  5. (Optional) Manually inspect dataset.
  6. Train classifier.
  7. Evaluate classifier.
  8. Export classification results as JSON.
  9. (Optional) Identify potentially mislabeled images.
* [Label Specification Syntax](#label-specification-syntax)
  * [CSV](#csv)
  * [JSON](#json)


# Overview

This README describes how to train and run an animal "species" classifier. "Species" is in quotes, because the classifier can be trained to identify animals at arbitrary levels within the biological taxonomy of animals.

This guide is written for internal use at Microsoft AI for Earth. Certain services, such as MegaDB and various private repos are only accessible interally within Microsoft. However, this guide may still be of interest to more technical users of the AI for Earth Camera Trap services.


# Setup

## Installation

Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html). Then create the conda environment using the following command. If you want to run PyTorch on a GPU, be sure to comment out `cpuonly` and uncomment `cudatoolkit` in `environment-classifier.yml`. If you need to add/remove/modify packages, make the appropriate change in the `environment-classifier.yml` file and run the following command again.

```bash
conda env update -f environment-classifier.yml --prune
```

Verify that *Pillow-SIMD* (installed from PyPI) overshadows the normal *Pillow* package (installed from conda). Open up a Python interpreter and run the following code. Make sure that the *Pillow* version ends in `'.postX'`, which indicates *Pillow-SIMD*.

```python
import PIL
print(PIL.__version__)
```

If this is running on a VM, enable remote Jupyter notebook access by doing the following. For more information, see the [Jupyter notebook server guide](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html).

1. Make sure that the desired port (e.g., 8888) is publicly exposed on the VM.
2. Run the following command to create a Jupyter config file at `$HOME/.jupyter/jupyter_notebook_config.py`.

    ```bash
    jupyter notebook --generate-config
    ```

3. Add the following line to the config file:

    ```python
    c.NotebookApp.ip = '*'
    ```

To use the *tqdm* widget in a notebook through JupyterLab (`jupyter lab`), make sure you have node.js installed, then run the following command. See the [*ipywidgets* installation guide](https://ipywidgets.readthedocs.io/en/latest/user_install.html) for more details.

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
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


# Running a classifier on new images

## 1. Run MegaDetector

Run MegaDetector on the new images to get an output JSON file in the format of the Batch API. MegaDetector can be run either locally or via the Batch API.

<details>
    <summary>Basic instructions for running MegaDetector locally</summary>

We assume that the images are in a local folder `/path/to/images`. Use [AzCopy](http://aka.ms/azcopy) if necessary to download the images from Azure Blob Storage.

From the CameraTraps repo folder, run the following. On a fast GPU, this should process ~3 images per second.

```bash
# Download the MegaDetector model file
wget -O md_v4.1.0.pb https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# install TensorFlow v1 and other dependences
conda env update -f environment-detector.yml --prune
conda activate cameratraps-detector

# run MegaDetector
python detection/run_tf_detector_batch.py md_v4.1.0.pb /path/to/images detections.json --recursive --output_relative_filenames
```

For more details, consult the [MegaDetector README](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md).
</details>


<details>
    <summary>Instructions for running MegaDetector via Batch API</summary>

See [`api/batch_processing/data_preparation/manage_api_submission.py`](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/data_preparation/manage_api_submission.py).
</details>


## 2. Crop images

Run `crop_detections.py` to crop the bounding boxes according to the detections JSON. Pass in an Azure Blob Storage container URL if the images are not stored locally and the detections were obtained from the Batch API. The crops are saved to `/path/to/crops`. Unless you have a good reason not to, use the `--square-crops` flag, which crops the tightest square enclosing each bounding box (which may have an arbitrary aspect ratio).

```bash
python crop_detections.py \
    detections.json \
    /path/to/crops \
    --images-dir /path/to/images \
    --container-url "https://account.blob.core.windows.net/container?sas_token" \
    --detector-version "4.1" \
    --confidence-threshold 0.8 \
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

Use the [label specification syntax](#label-specification-syntax) to specify the taxons and/or dataset classes that constitute each target category. If using the CSV format, convert it to the JSON specification syntax using `python csv_to_json.py`.

**Build a mapping from desired target categories to MegaClassifier labels.**

Run the `map_classification_categories.py` script with the target label specification JSON to create a mapping from target categories to MegaClassifier labels. The output file is another JSON file representing a dictionary whose keys are target categories and whose values are lists of MegaClassifier labels. MegaClassifier labels who are not explictly assigned a target are assigned to a target named "other". Each MegaClassifier label is assigned to exactly one target category.

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

Finally, merge the classification results CSV with the original detection JSON file. Use the `--threshold` argument to exclude predicted categories from the JSON file if their confidence is below a certain threshold. This file can then be opened in Timelapse (requires v2.2.3.7.1 or greater).

```bash
```

# Typical Training Pipeline

## 1. Select classification labels for training.

Create a classification labels specification JSON file (usually named `label_spec.json`). This file defines the labels that our classifier will be trained to distinguish, as well as the original dataset labels and/or biological taxons that will map to each classification label. See the required format [here](#json).

For MegaClassifier, see `megaclassifier_label_spec.ipynb` to see how the label specification JSON file is generated.

For bespoke classifiers, it is likely easier to write a CSV file instead of manually writing the JSON file. We then translate to JSON using `csv_to_json.py`. The CSV syntax can be found [here](#csv).


## 2. Validate the classification labels specification JSON file, and generate a list of matching images.

In `json_validator.py`, we validate the classification labels specification JSON file. It checks that the specified taxa are included in the master taxonomy CSV file, which specifies the biological taxonomy for every dataset label in MegaDB. The script then queries MegaDB to list all images that match the classification labels specification, and optionally verifies that each image is only assigned a single classification label.

TODO: explain `-m` flag for mislabeled images.

The output of `json_validator.py` is another JSON file (`queried_images.json`) that maps image names to a dictionary of properties:

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
    --json-indent 1 \
    -m /path/to/classifier-training/megab_mislabeled
```


## 3. Generate bounding boxes using MegaDetector.

While some labeled images in MegaDB already have ground-truth bounding boxes, other images do not. For the labeled images without bounding box annotations, we run MegaDetector to get bounding boxes. MegaDetector can be run either locally or via the Batch Detection API.

This step consists of 3 sub-steps:
1. Run MegaDetector (either locally or via Batch API) on the queried images.
2. Cache MegaDetector results on the images to JSON files in `classifier-training/mdcache`.
3. Download and crop the images to be used for training the classifier.

<details>
    <summary>To run MegaDetector locally</summary>
    Not implemented yet.
</details>

<details>
    <summary>To use the MegaDetector Batch Detection API</summary>

We use the `detect_and_crop.py` script. In theory, we can do everything we need in a single invocation. The script groups the queried images by dataset and then submits 1 "task" to the Batch Detection API for each dataset. It knows to wait for each task to finish running, before starting to download and crop the images based on bounding boxes. Because of the resume file, in theory it should be OK to cancel the script once the tasks are all submitted, then re-run the script later with the exact same arguments to fetch the results and begin the downloading and cropping.

```bash
python detect_and_crop.py \
    $BASE_LOGDIR/queried_images.json \
    $BASE_LOGDIR \
    -c /path/to/classifier-training/mdcache -v "4.1" \
    -d batchapi -r $BASE_LOGDIR/resume.json \
    -p /path/to/crops --square-crops -t 0.9 -n 50 \
    --save-full-images -i /path/to/images
```

However, because the Batch Detection API often returns incorrect responses, in practice we often need to call `detect_and_crop.py` multiple times. It is important to understand the 2 different "modes" of the script.

1. Call the Batch Detection API, and cache the results.
    * To run this mode: set `--detector batchapi`
    * To skip this mode: set `--detector skip`
2. Using ground truth and cached detections, crop the images.
    * To run this mode: set `--cropped-images-dir /path/to/crops`
    * To skip this mode: don't set `--cropped-images-dir`

Thus, we will first call the Batch Detection API. This will save a `resume.json` file that contains all of the task IDs. Because the Batch Detection API does not always respond with the correct task status, the only real way to verify if a task has finished running is to check the `async-api-*` Azure Storage container and see if the output files are there.

```bash
python detect_and_crop.py \
    $BASE_LOGDIR/queried_images.json \
    $BASE_LOGDIR \
    -c /path/to/classifier-training/mdcache -v "4.1" \
    -d batchapi -r $BASE_LOGDIR/resume.json
```

When a task finishes running, manually create a JSON file for each task according to the [Batch Detection API response format](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#api-outputs). Save the JSON file to `$BASE_LOGDIR/batchapi_response/dataset.json`. Then, use `cache_batchapi_outputs.py` to cache these results:

```bash
python cache_batchapi_outputs.py \
    $BASE_LOGDIR/batchapi_response/dataset.json \
    --dataset dataset \
    -c $HOME/classifier-training/mdcache -v 4.1
```

Finally, we download and crop the images based on the ground truth and detected bounding boxes. On a VM, expect this download and cropping step to run at ~60 images per second (~5 hours for 1 million images).

```bash
python detect_and_crop.py \
    $BASE_LOGDIR/queried_images.json \
    $BASE_LOGDIR \
    -c /path/to/classifier-training/mdcache -v "4.1" \
    -d skip -p /path/to/crops --square-crops -t 0.9 -n 50 \
    --save-full-images -i /path/to/images
```
</details>


## 4. Create classification dataset and split into train/val/test sets by location.

```bash
python create_classification_dataset.py \
    BASE_LOGDIR \
    --mode csv splits \
    -q $BASE_LOGDIR/queried_images.json \
    -c /path/to/crops \
    -d /path/to/classifier-training/mdcache -v "4.1" \
    -t 0.8 --min-locs 20 \
    --val-frac 0.2 --test-frac 0.2 \
    --method random
```

## 5. (Optional) Manually inspect dataset.

Copy the `inspect_dataset.ipynb` notebook into `$BASE_LOGDIR`. Open a Jupyter lab or notebook instance, and run the notebook.

## 6. Train classifier.

```bash
python train_classifier.py \
    $BASE_LOGDIR \
    /path/to/crops_sq \
    -m efficientnet-b3 --pretrained
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
* `--weight-decay 1e-6`

## 7. Evaluate classifier.

```bash
python evaluate_model.py $BASE_LOGDIR/$LOGDIR ckpt_XX.pt
```

## 8. Export classification results as JSON.

Once we have the `output_{split}.csv.gz` files, we can export our classification results in the Batch Detection API JSON format. The following command generates such a JSON file for the images from the test set, including only classification probabilities greater than 0.1, and also including the true label:

```bash
python merge_classification_detection_output.py \
    $BASE_LOGDIR/$LOGDIR/outputs_test.csv.gz \
    $BASE_LOGDIR/label_index.json \
    $BASE_LOGDIR/queried_images.json \
    -n "<classifier_name>" \
    -c $HOME/classifier-training/mdcache -v "4.1" \
    -o $BASE_LOGDIR/$LOGDIR/outputs_test.json \
    --label last -t 0.1
```

## 9. (Optional) Identify potentially mislabeled images.

We can now use our trained classifier to identify potentially mislabeled images by looking at the model's false positives. A "mislabeled candidate" is defined as an image meeting both of the following criteria:
- according to the ground-truth label, the model made an incorrect prediction
- the model's prediction confidence exceeds its confidence for the ground-truth label by some minimum confidence.

At this point, we should have the following folder structure:
```
BASE_LOGDIR/
    queried_images.json           # generated in step (?)
    label_index.json              # generated in step (?)
    LOGDIR/                       # generated in step (?)
        outputs_{split}.csv.json  # generated in step (7)
```

We generate a JSON file that can be loaded into Timelapse to help us review mislabeled candidates. We again use `merge_classification_detection_output.py`. However, instead of outputting raw classification probabilities, we output the margin of error by passing the `--relative-conf` flag.

```bash
python merge_classification_detection_output.py $BASE_LOGDIR/$LOGDIR/outputs_test.csv.gz \
    $BASE_LOGDIR/label_index.json \
    $BASE_LOGDIR/queried_images.json \
    -n "myclassifier"
    -c $HOME/classifier-training/mdcache -v "4.1"
    -o $BASE_LOGDIR/$LOGDIR/outputs_json_test_set_relative_conf.json --relative-conf
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

Until AzCopy improves its performance for its undocumented `--list-of-files` option, its performance is generally much slower. However, we can use it as follows:

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
python save_mislabeled.py $HOME/classifier-training /path/to/mislabeled_images.csv
```


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

# exclude certain rows or taxons
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
            "idfg_swwlf_2019": ["elk", "muledeer", "whitetaileddeer"],
        },

        "max_count": 50000  // only include up to this many images (not crops)

        // prioritize images from certain datasets over others,
        // only used if "max_count" is given
        "prioritize": [
            ["idfg_swwlf_2019"],  // give 1st priority to images from this list of datasets
            ["idfg"]  // give 2nd priority to images from this list of datasets
            // give remaining priority to images from all other datasets
        ],

    },

    // name of another classification label
    "bird": {
        "taxa": [
            {
                "level": "class",
                "name": "aves",
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
