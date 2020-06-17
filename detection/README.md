# Detection

This folder contains scripts and configuration files for training and evaluating a detector for objects of classes `animal`, `person` and `vehicle`, and two scripts to use the model to perform inference on incoming images. 

We use the TensorFlow Object Detection API ([TFODAPI](https://github.com/tensorflow/models/tree/master/research/object_detection)) for model training with TensorFlow 1.12.0. See the Dockerfile used for training [here](https://github.com/microsoft/ai4eutils/blob/master/TF_OD_API/Dockerfile_TFODAPI).

Bounding boxes predicted by the detector are in normalized coordinates, as `[ymin, xmin, ymax, xmax]`, with the origin in the upper-left of the image. This is different from 
- the COCO Camera Trap format, which uses absolute coordinates in `[xmin, ymin, width_of_box, height_of_box]` (see [data_management](../data_management/README.md))
- the batch processing API's output and entries in the MegaDB, which use normalized coordinates in `[xmin, ymin, width_of_box, height_of_box]` (see [batch_processing](../api/batch_processing#detector-outputs))


## Content

- `detector_training/model_main.py`: a modified copy of the entry point script for training the detector, taken from [TFODAPI](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py).

- `detector_training/experiments/`: a folder for storing the model configuration files defining the architecture and (loosely, since learning rate is often adjusted manually) the training scheme. Each new detector project or update is in a subfolder, which could contain a number of folders for various experiments done for that project/update. Not every run's configuration file needs to be recorded here (e.g. adjusting learning rate, new starting checkpoint), since TFODAPI copies `pipeline.config` to the model output folder at the beginning of the run; the configuration files here capture high-level info such as model architecture. 

- `detector_eval/`: scripts for evaluating various aspects of the detector's performance. We use to evaluate test set images stored in TF records, but now we use the batch processing API to process the test set images stored individually in blob storage. This is more parallelizable. Functions in `detector_eval.py` works with the API's output format and the ground truth format from querying the MegaDB (described below).

- `run_tf_detector.py`: the simplest demonstration of how to invoke a TFODAPI-trained detector.

- `run_tf_detector_batch.py`: runs the detector on a collection images; output is the same as that produced by the batch processing API.


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

In any script or notebook, give each entry a random ID, and append it to the name of the dataset this image comes from, separated by a `+`:

```python
import uuid

for i in entries:
    i['download_id'] = '{}+{}'.format(i['dataset'], uuid.uuid4())
```

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
 "download_id": "dataset_name+8896d576-3f14-11ea-b3bb-9801a5a664ab",
 "new_entry": true
}
```

### Download the images

Use `data_management/megadb/download_images.py` to download the new images, probably to an attached disk so that there is enough space. Use the flag `--only_new_images` if in the above step you added the `new_entry` field to images that still need to be downloaded. 


### Split the images into train/val/test set

Use `data_management/megadb/split_images.py`. It will move the images to new folders, so be careful. It will look up the splits in the Splits table in MegaDB, and any entries that do not have a location field will be placed in the training set.


### Create TFrecords

Use `data_management/tfrecords/make_tf_records_megadb.py`. The class mappings are defined towards the top of this script.


Deprecated: `data_management/tfrecords/make_tfrecords.py` takes in your CCT format json database, creates an intermediate json conforming to the format that the resulting tfrecords require, and then creates the tfrecords. Images are split by `location` according to `split_frac` that you specify in the `Configurations` section of the script.

If you run into issues with corrupted .jpg files, you can use `database_tools/remove_corrupted_images_from_database.py` to create a copy of your database without the images that TensorFlow cannot read. You don't need this step; `make_tfrecords.py` will ignore any images it cannot read.


### Install TFODAPI

TFODAPI requires TensorFlow >= 1.9.0

To set up a stable version of TFODAPI, which is a project in constant development, we use the `Dockerfile` and set-up script (checking out a specific commit from the TFODAPI repo) in our utilities repo at https://github.com/Microsoft/ai4eutils/tree/master/TF_OD_API.

Alternatively, follow [installation instructions for TFODAPI](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
 
If you are having protobuf errors, install protocol buffers from binary as described [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)



## Set up an experiment
- Create a directory in the `detector_training/experiments` folder of this section to keep a record of the model configuration used.

- Decide on architecture
    - Our example uses Faster R-CNN with Inception ResNet V2 backbone and atrous convolutions. 

- Download appropriate pre-trained model from the [TFODAPI model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
    - Our example pre-trained model is at `models/object_detection/faster_rcnn_inception_resnet_v2_atrous/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28`
    - You can also start training from a pre-trained TensorFlow classification model (instructions from a while ago)
        - We have an example of this in the fileshare at `models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_with_ss_pretrained_backbone/`
        - Note that to start from a classification backbone, you must add `from_detection_checkpoint: false` to the `train_config` section in your config file
        - To train a classification backbone, we recommend using [this visipedia classification code base](https://github.com/visipedia/tf_classification)
        
        - Copy the sample config (typically `pipeline.config`) for that architecture to your experiment folder as a starting point, either from the samples in the TFODAPI repo or from the folder you downloaded containing the pre-trained model

- Modify the config file to point to locations of your training and validation tfrecords, and pre-trained model

- Make any other config changes you want for this experiment (learning rate, data augmentation, etc). Try visualizing only a handful of images during evaluation (~20) because visualizing ~200 can result in a 30GB large TensorBoard events file at the end of training a MegaDetector.

  
## Start training

On a VM with TFODAPI set-up, run training in a tmux session (inside a Docker container or otherwise). 

Example command to start training:

```bash
python model_main.py \
--pipeline_config_path=/experiment1/run1/pipeline.config \
--model_dir=/experiment1/run1_out/ \
--sample_1_of_n_eval_examples 10
```

You can sample more of the validation set (set `sample_1_of_n_eval_examples` to a smaller number); in that case, evaluate less often by changing `save_checkpoints_steps` in `model_main.py` (flags may not work - change the code directly).

Alternatively, use Azure Machine Learning (AML) to run the experiment. Notebook showing how to use the AML Python SDK to run TFODAPI experiments: `detector_training/aml_mdv4.ipynb`.


## Watch training on TensorBoard
Make sure that the desired port (port `6006` in this example) is open on the VM, and that you're in a tmux session.

Example command:

```bash
tensorboard --logdir run1:/experiment1/run1_out/,run2:/experiment1/run1_out_continued/ --port 6006 --window_title "experiment1 both runs"
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
