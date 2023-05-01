# Overview

This folder contains scripts and configuration files for training and evaluating [MegaDetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).  If you are looking to <b>use</b> MegaDetector, you probably don't want to start with this page; instead, start with the [MegaDetector page](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).  If you are looking to fine-tune MegaDetector on new data, you also don't want to start with this page; instead, start with the [YOLOv5 training guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

# Format notes

Bounding boxes predicted by MegaDetector are in normalized coordinates, as `[ymin, xmin, ymax, xmax]`, with the origin in the upper-left of the image. This is different from 
- the COCO Camera Trap format, which uses absolute coordinates in `[xmin, ymin, width_of_box, height_of_box]` (see [data_management](../data_management/README.md))
- the batch processing API's output and entries in the MegaDB, which use normalized coordinates in `[xmin, ymin, width_of_box, height_of_box]` (see [batch_processing](../api/batch_processing#detector-outputs))

# Contents of this folder

- `detector_training/experiments/`: a folder for storing the model configuration files defining the architecture and (loosely, since learning rate is often adjusted manually) the training scheme. Each new detector project or update is in a subfolder, which could contain a number of folders for various experiments done for that project/update. Not every run's configuration file needs to be recorded here (e.g. adjusting learning rate, new starting checkpoint), since TFODAPI copies `pipeline.config` to the model output folder at the beginning of the run; the configuration files here record high-level info such as model architecture. 

- `detector_eval/`: scripts for evaluating various aspects of the detector's performance. We use to evaluate test set images stored in TF records, but now we use the batch processing API to process the test set images stored individually in blob storage. Functions in `detector_eval.py` works with the API's output format and the ground truth format from querying the MegaDB (described below).

- `run_detector.py`: the simplest demonstration of how to invoke a detector.

- `run_detector_batch.py`: runs the detector on a collection images; output format is documented [here](https://github.com/microsoft/CameraTraps/tree/main/api/batch_processing/#batch-processing-api-output-format).

# Training MegaDetector

## Assembling the training data set

These steps document the steps taken to assemble the training data set when MDv5 was trained at Microsoft; this section is not meaningful outside of Microsoft.

### Query MegaDB for the images of interest

Use `data_management/megadb/query_script.py` to query for all desired image entries. Write the query to use or select one from the examples at the top of the script. You may need to adjust the code parsing the query's output if you are using a new query. Fill in the output directory and other parameters also near the top of the script. 

To get labels for training  MegaDetector, use the query `query_bbox`. Note that to include images that were sent for annotation and were confirmed to be empty, make sure to specify `ARRAY_LENGTH(im.bbox) >= 0` to include the ones whose `bbox` field is an empty list. 

Running this query will take about 10+ minutes; this is a relatively small query so no need to increase the throughput of the database. The output is a JSON file containing a list, where each entry is the label for an image:
 
```json
{
 "bbox": [
  {
   "category": "person",
   "bbox": [
    0.3023,
    0.487,
    0.5894,
    0.4792
   ]
  }
 ],
 "file": "Day/1/IMAG0773 (4).JPG",
 "dataset": "dataset_name",
 "location": "location_designation"
}
```

### Assign each image a `download_id`

To avoid creating nested directories for downloaded images, we give each image a `download_id` to use as the file name to save the image at.

In any script or notebook, give each entry a unique ID (`<dataset>.seq<seq_id>.frame<frame_num>`).

If you are preparing data to add to an existing, already downloaded collection, add a field `new_entry` to the entry.

Save this version of the JSON list:

```json
{
 "bbox": [
  {
   "category": "person",
   "bbox": [
    0.3023,
    0.487,
    0.5894,
    0.4792
   ]
  }
 ],
 "file": "Day/1/IMAG0773 (4).JPG",
 "dataset": "dataset_name",
 "location": "location_designation",
 "download_id": "tnc_islands.seqab350628-ff22-2a29-8efa-boa24db24b57.frame0",
 "new_entry": true
}
```

### Download the images

Use `data_management/megadb/download_images.py` to download the new images, probably to an attached disk. Use the flag `--only_new_images` if in the above step you added the `new_entry` field to images that still need to be downloaded. 


### Split the images into train/val/test set

Use `data_management/megadb/split_images.py` to move the images to new folders `train`, `val`, and `test`. It will look up the splits in the Splits table in MegaDB, and any entries that do not have a location field will be placed in the training set.


## Training with YOLOv5

This section documents the environment in which MegaDetector v5 was trained; for more information about these parameters, see the [YOLOv5 training guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

With image size 1280px, starting with pre-trained weights (automatically downloaded from latest release) of the largest model (yolov5x6.pt). Saving checkpoint every epoch. Example:

```
export WANDB_CACHE_DIR=/camtraps/wandb_cache

docker pull nvidia/cuda:11.4.2-runtime-ubuntu20.04

(or yasiyu.azurecr.io/yolov5_training with the YOLOv5 repo dependencies installed)

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -d -it -v /marmot_disk_0/camtraps:/camtraps nvcr.io/nvidia/pytorch:21.10-py3 /bin/bash 

torchrun --standalone --nnodes=1 --nproc_per_node 2 train.py --project megadetectorv5 --name camonly_mosaic_xlarge_dist_5 --noval --save-period 1 --device 0,1 --batch 8 --imgsz 1280 --epochs 10 --weights yolov5x6.pt --data /home/ilipika/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5_yolo/data_camtrap_images_only.yml --hyp /home/ilipika/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5_yolo/hyp_mosaic.yml
```
