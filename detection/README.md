# Detection

This folder contains scripts and configuration files for training and evaluating a detector for objects of classes `animal`, `person` and `vehicle`, and two inference scripts to apply the model on new images. 

We use the TensorFlow Object Detection API ([TFODAPI](https://github.com/tensorflow/models/tree/master/research/object_detection)) for model training with TensorFlow 2.2.0, Python 3.6.9 (prior to MegaDetector v5, we used TF 1.12.0 with this older [Dockerfile](https://github.com/microsoft/ai4eutils/blob/master/TF_OD_API/Dockerfile_TFODAPI).).

Bounding boxes predicted by the detector are in normalized coordinates, as `[ymin, xmin, ymax, xmax]`, with the origin in the upper-left of the image. This is different from 
- the COCO Camera Trap format, which uses absolute coordinates in `[xmin, ymin, width_of_box, height_of_box]` (see [data_management](../data_management/README.md))
- the batch processing API's output and entries in the MegaDB, which use normalized coordinates in `[xmin, ymin, width_of_box, height_of_box]` (see [batch_processing](../api/batch_processing#detector-outputs))


## Content

- `detector_training/experiments/`: a folder for storing the model configuration files defining the architecture and (loosely, since learning rate is often adjusted manually) the training scheme. Each new detector project or update is in a subfolder, which could contain a number of folders for various experiments done for that project/update. Not every run's configuration file needs to be recorded here (e.g. adjusting learning rate, new starting checkpoint), since TFODAPI copies `pipeline.config` to the model output folder at the beginning of the run; the configuration files here record high-level info such as model architecture. 

- `detector_eval/`: scripts for evaluating various aspects of the detector's performance. We use to evaluate test set images stored in TF records, but now we use the batch processing API to process the test set images stored individually in blob storage. Functions in `detector_eval.py` works with the API's output format and the ground truth format from querying the MegaDB (described below).

- `run_tf_detector.py`: the simplest demonstration of how to invoke a TFODAPI-trained detector (to be updated for TF2).

- `run_tf_detector_batch.py`: runs the detector on a collection images; output is the same as that produced by the batch processing API (to be updated for TF2).


## Steps in a detection project

### Query MeagDB for the images of interest

Use `data_management/megadb/query_script.py` to query for all desired image entries. Write the query to use or select one from the examples at the top of the script. You may need to adjust the code parsing the query's output if you are using a new query. Fill in the output directory and other parameters also near the top of the script. 

To get labels for training the MegaDetector, use the query `query_bbox`. Note that to include images that were sent for annotation and were confirmed to be empty by the annotators (iMerit), make sure to specify `ARRAY_LENGTH(im.bbox) >= 0` to include the ones whose `bbox` field is an empty list. 

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


### Create TFrecords

Use `data_management/tfrecords/make_tfrecords_megadb.py`. The class mappings are defined towards the top of this script.

Deprecated: `data_management/tfrecords/make_tfrecords.py` takes in your CCT format json database, creates an intermediate json conforming to the format that the resulting tfrecords require, and then creates the tfrecords. Images are split by `location` according to `split_frac` that you specify in the `Configurations` section of the script.

If you run into issues with corrupted .jpg files, you can use `database_tools/remove_corrupted_images_from_database.py` to create a copy of your database without the images that TensorFlow cannot read. You don't need this step; `make_tfrecords.py` will ignore any images it cannot read.


### Install TFODAPI

Follow these [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) to install the TF Object Detection API.

## Set up an experiment
- Create a directory in the `detector_training/experiments` folder of this section to keep a record of the model configuration used.

- Decide on architecture and pre-trained model.

- Download appropriate pre-trained model from the [TFODAPI model zoo TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
    - MegaDetectors prior to v5 used `faster_rcnn_inception_resnet_v2_atrous`.
        
- Copy the sample config (typically `pipeline.config`) for that architecture to your experiment folder as a starting point, either from the samples in the TFODAPI repo or from the folder you downloaded containing the pre-trained model.

- Modify the config file to point to locations of your training and validation tfrecords, and pre-trained model. We do not modify `num_classes` to use all pre-trained weights.

- Make any other config changes you want for this experiment (learning rate, data augmentation etc). Try visualizing only a handful of images during evaluation (~20) because visualizing ~200 can result in a 30GB large TensorBoard events file at the end of training.

  
## Start training

On a VM with TFODAPI set-up, run training in a tmux session (inside a Docker container or otherwise). 

First copy the configuration file in the mirrored repo on the VM to the model training folder on the disk, then call the `model_main_tf2.py` script:
```
MODEL_DIR=0921_effid3_lr6e-4

cp /home/mongoose/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5/pipeline_efficientdet_d3.config /mongoose_disk_0/camtraps/mdv5/${MODEL_DIR}/pipeline_efficientdet_d3.config

python model_main_tf2.py \
--pipeline_config_path=/mongoose_disk_0/camtraps/mdv5/${MODEL_DIR}/pipeline_efficientdet_d3.config \
--model_dir=/mongoose_disk_0/camtraps/mdv5/${MODEL_DIR} \
--checkpoint_every_n 20000 \
--num_train_steps 1000000000 \
--alsologtostderr
```

Leaving the `num_workers` parameter as 1 will use the `tf.distribute.MirroredStrategy`, supporting synchronous distributed training on multiple GPUs on one machine

In another process, run the evaluation job but not on a GPU:
```
CUDA_VISIBLE_DEVICES=-1 

python model_main_tf2.py \
--pipeline_config_path=/mongoose_disk_0/camtraps/mdv5/${MODEL_DIR}/pipeline_efficientdet_d3.config \
--model_dir=/mongoose_disk_0/camtraps/mdv5/${MODEL_DIR} \
--checkpoint_dir=/mongoose_disk_0/camtraps/mdv5/${MODEL_DIR} \
--alsologtostderr
```


## Watch training on TensorBoard
Make sure that the desired port (port `6006` in this example) is open on the VM, and that you're in a tmux session.

Example command:

```bash
tensorboard --logdir . --port 6006 --window_title "MDv5 effid3" --bind_all
```

## Export best model

Use the TFODAPI's `export_inference_graph.py` ([documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)) to export a model based on a checkpoint of your choice (e.g. best one according to validation set mAP@0.5IoU).

Example call:

```bash
python models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=/experiment1/run1/pipeline.config \
    --trained_checkpoint_prefix=/experiment1/run1_out/model.ckpt-141004 \
    --output_directory=/experiment1/run1_model_141004/
```


## Run inference on test sets - deprecated

(As explained above, we now register the model with the AML workspace that the batch processing API is using, and score test set images stored individually in blob storage in parallel)

Run TFODAPI's `inference/infer_detections.py` using the exported model, on tfrecords of the (test) set

Example call:

```bash
cd /lib/tf/models/research/object_detection/inference/

TF_RECORD_ROOT=/ai4edevshare/tfrecords/megadetector_v3
TF_RECORD_FILES=$(ls -1 ${TF_RECORD_ROOT}/???????~val__-?????-of-????? | tr '\n' ',')

python infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=/megadetectorv3/eval/megadetector_v2_on_val_v3.tfrecord \
  --inference_graph=/ai4edevshare/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector_v2/frozen_inference_graph.pb \
  --discard_image_pixels
```
Then you can use `data_management/tfrecords/tools/read_from_tfrecords.py` to read detection results from the output tfrecord into python dicts and save them as a pickle file.


## Evaluation

See `detector_eval`. We usually start a notebook to produce the visualizations, using functions in `detector_eval/detector_eval.py`.


## Using YOLO v5

With image size 1280px, starting with pre-trained weights (automatically downloaded from latest release) of the largest model (yolov5x6.pt). Saving checkpoint every epoch. Example:

```
export WANDB_CACHE_DIR=/camtraps/wandb_cache

docker pull nvidia/cuda:11.4.2-runtime-ubuntu20.04

(or yasiyu.azurecr.io/yolov5_training with the YOLOv5 repo dependencies installed)

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -d -it -v /marmot_disk_0/camtraps:/camtraps nvcr.io/nvidia/pytorch:21.10-py3 /bin/bash 


torchrun --standalone --nnodes=1 --nproc_per_node 2 train.py --project megadetectorv5 --name camonly_mosaic_xlarge_dist_5 --noval --save-period 1 --device 0,1 --batch 8 --imgsz 1280 --epochs 10 --weights yolov5x6.pt --data /home/ilipika/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5_yolo/data_camtrap_images_only.yml --hyp /home/ilipika/camtraps/pycharm/detection/detector_training/experiments/megadetector_v5_yolo/hyp_mosaic.yml
```
