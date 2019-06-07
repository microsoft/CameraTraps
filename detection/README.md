# Detection

This folder contains scripts and configuration files for training and evaluating a detector (generic in terms of ecosystems and species) for objects of classes `animal`, `person` and `vehicle`. 

We use the TensorFlow Object Detection API ([TFODAPI](https://github.com/tensorflow/models/tree/master/research/object_detection)) for model training. 

Bounding boxes predicted by the detector are represented in normalized coordinates, as `[ymin, xmin, ymax, xmax]`, with the origin in the upper-left of the image. This is different from the COCO Camera Trap format of our json databases, which uses absolute coordinates in `[x, y, width_of_box, height_of_box]` (see [data_management](api/detector_batch_processing/README.md)), and the batch processing API also converts them to relative coordinates in `[x, y, width_of_box, height_of_box]`.


- `detection_training/model_main.py`: a modified copy of the entry point script for training the detector, taken from [TFODAPI](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py).

- `detection_training/experiments/`: a folder for storing the model configuration files defining the architecture and (loosely, since learning rate is often adjusted manually) the training scheme. Each new detector project or update is in a subfolder, which could contain a number of folders for various experiments done for that project/update. Not every run's configuration file needs to be recorded here (e.g. adjusting learning rate, new starting checkpoint), since TFODAPI copies `pipeline.config` to the model output folder at the beginning of the run; the configuration files here capture high-level info such as model architecture. 

- `detection_eval/`: scripts for evaluating various aspects of the detector's performance. To evaluate such a detector, first run TFODAPI's `inference/infer_detections.py` using a frozen inference graph based on the checkpoint to evaluate, on tfrecords of the (validation) set. This produces a tfrecord containing all the detection results of the (validation) examples. Then use [data_management/tfrecords/tools/read_from_tfrecords.py](data_management/tfrecords/tools/read_from_tfrecords.py) to extract the info from this tfrecord into a pickled json (`.p` file), which would be the input to all scripts in the `detection_eval` folder. 
    - In the future, we will adapt these scripts to work on output format of the batch processing API as well to easily evaluate against images not in tfrecords.

- `run_tf_detector.py`: the simplest demonstration of how to invoke a TFODAPI-trained detector.

- `run_tf_detector_batch.py`: runs the detector on a collection images; output is the same as that produced by the batch processing API.


# Steps in a detection project


## Create a COCO Camera Traps (CCT) format json database for your data
Use code from `data_management`.


## Install TFODAPI

TFODAPI requires TensorFlow >= 1.9.0

To set up a stable version of TFODAPI, which is a project in constant development, we use the `Dockerfile` and set-up script in our utilities repo at https://github.com/Microsoft/ai4eutils/tree/master/TF_OD_API.

Alternatively, follow [installation instructions for TFODAPI](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
 
If you are having protobuf errors, install protocol buffers from binary as described [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)


## Create tfrecords
Use code in `data_management/tfrecords` to create tfrecord files for your data.

- `make_tfrecords.py` takes in your CCT format json database, creates an intermediate json conforming to the format that the resulting tfrecords require, and then creates the tfrecords. Images are split by `location` according to `split_frac` that you specify in the `Configurations` section of the script.
 
- If you run into issues with corrupted .jpg files, you can use `database_tools/remove_corrupted_images_from_database.py` to create a copy of your database without the images that TensorFlow cannot read. You don't need this step if you don't want to exclude corrupted images from the database, as `make_tfrecords.py` will ignore any images it cannot read.


## Set up an experiment
- Create a directory in the `detection_training/experiments` folder of this section to keep a record of the model configuration used.

- Decide on architecture
    - Our example uses Faster R-CNN with Inception ResNet V2 backbone and Atrous Convolutions. 

- Download appropriate pre-trained model from the [TFODAPI model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
    - Our example pre-trained model is at `models/object_detection/faster_rcnn_inception_resnet_v2_atrous/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28`
    - You can also start training from a pre-trained TensorFlow classification model
    
    - We have an example of this in the fileshare at `models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_with_ss_pretrained_backbone/`
    - Note that to start from a classification backbone, you must add `from_detection_checkpoint: false` to the `train_config` section in your config file
    - To train a classification backbone, we recommend using [this visipedia classification code base](https://github.com/visipedia/tf_classification)
    
    - Copy the sample config (typically `pipeline.config`) for that architecture to your experiment folder as a starting point, either from the samples in the TFODAPI repo or from the folder you downloaded containing the pre-trained model

    - Modify the config file to point to locations of your training tfrecords, eval tfrecords, and pre-trained model

    - Make any other config changes you want for this experiment (learning rate, data augmentation, etc)

  
## Start training

On a VM with TFODAPI set-up, run training in a tmux session (inside a Docker container or otherwise). 

Example command to start training:
```
python model_main.py \
--pipeline_config_path=/experiment1/run1/pipeline.config \
--model_dir=/experiment1/run1_out/ \
--sample_1_of_n_eval_examples 10
```


## Watch training on TensorBoard
Make sure that the desired port (port `6006` in this example) is open on the VM, and that you're in a tmux session.

Example command:
```
tensorboard --logdir run1:/experiment1/run1_out/,run2:/experiment1/run1_out_continued/ --port 6006 --window_title "experiment1 both runs"
```

## Export best model

Use the TFODAPI's `export_inference_graph.py` ([documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)) to export a model based on a checkpoint of your choice (e.g. best one according to validation set mAP@0.5IoU).

Example call:
```
python models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=/experiment1/run1/pipeline.config \
    --trained_checkpoint_prefix=/experiment1/run1_out/model.ckpt-141004 \
    --output_directory=/experiment1/run1_model_141004/
```


## Run inference on test sets

Run TFODAPI's `inference/infer_detections.py` using the exported model, on tfrecords of the (test) set

Example call:
```
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


# Evaluation
`detection_eval` contains code for evaluating the detection results, based on the pickle file obtained in the previous step.

Evaluation scripts provided will include evaluating at object, image, and sequence levels, evaluating detection models as classifiers (which allows you to evaluate on data that has only class-level annotations), evaluating models per-camera-location, and evaluating models per-species.

