# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Contents

This repo contains several different folders, organized by type of task:

## AirSim
Code for working with AirSim

## database_tools
Code for creating, visualizing stats, or editing COCO-CameraTraps style json databases

## detection_eval
Code for performing offline evaluation of detection results, with or without sequences

## annotation
Code for creating new annotation tasks and converting annotations to COCO-CameraTraps format

## pipeline
Bash scripts for training, exporting, and running inference on detectors

## tfrecords
Code for creating or reading from tfrecord files, based on:

https://github.com/visipedia/tfrecords

# To start a new detection project:

## Create COCO-CameraTraps style json database for your data
Use code from "database_tools" and/or "annotation"

## Follow installation instructions

TFODAPI works best in python 2.7.  tfrecords files require python 2.7.  
TFODAPI requires Tensorflow >= 1.9.0

Follow [installation instructions for TFODAPI](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
 
If you are having protobuf errors, install protocol buffers from binary as described [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Create tfrecords
Use code in tfrecords to create tfrecord files for your data

* make_tfrecords_from_json.py demonstrates how to use the provided functions to create one set of tfrecord files from a single .json file
* if you want to run oneclass detection, first convert yor json to oneclass using database_tools/make_oneclass_json.py 
* If you run into issues with corrupted .jpg files, you can use database_tools/remove_corrupted_images_from_database.py to create a copy of your database without the images that tensorflow cannot read 

## Set up experiment directory
* Create experiment directory
  * Within experiment directory create `configs` folder
  * Example can be seen in the ai4e fileshare at sample_object_detection_experiment_directory/
* Decide on architecture
  * Our example uses Faster-RCNN with Inception Resvet V2 backbone and Atrous Convolutions 
* Download appropriate pretrained model from [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  * Our example pretrained model is at models/object_detection/faster_rcnn_inception_resnet_v2_atrous/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
  * You can also start training from a pretrained tensorflow classification model
    * We have an example of this in the fileshare at models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_with_ss_pretrained_backbone/
    * Note that to start from a classification backbone, you must add "from_detection_checkpoint: false" to train_config{} in your config file
    * To train a classification backbone, we recommend using [this visipedia classification code base](https://github.com/visipedia/tf_classification)
* Copy tf sample config (typically pipeline.config) for that architecture to your configs folder as a starting point (find samples in tfmodels/research/object_detection/samples/configs/)
  * Point within config file to the locations of your training tfrecords, eval tfrecords, pretrained model
  * Make any other config changes you want for this experiment (learning rate, data augmentation, etc.) using the params described in [preprocessor.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) 
  * Our example can be seen in sample_object_detection_experiment_directory/configs/pipeline.config
  
## Run training

You can use the bash script pipeline/run_training.sh

Alternatively, you can run everything from the command line.

Example call:
```
python  tfmodels/research/object_detection/model_main.py \
--pipeline_config_path=/ai4efs/sample_object_detection_experiment_directory/configs/pipeline.config \
--model_dir=/ai4efs/sample_object_detection_experiment_directory/ \
--num_train_steps=200000 \
--num_eval_steps=1725 \
--alsologtostderr

```

## Watch training on tensorboard
Make sure you have port 6006 open on your VM

Example call:
```
tensorboard --logdir /ai4efs/sample_object_detection_experiment_directory/ --port 6006
```

## Export best model

Use pipeline/run_export_model.sh

Alternatively...

Example call:
```
python tfmodels/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=/ai4efs/sample_object_detection_experiment_directory/configs/pipeline.config \
    --trained_checkpoint_prefix=/ai4efs/sample_object_detection_experiment_directory/model.ckpt-[XXXXXX] \
    --output_directory=/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector_sample/

```
## Run inference on test sets

Use pipeline/run_inference_all.sh to extract bboxes for all test sets to python dicts, or use the individual bash scripts for individual test sets

Alternatively...

Example call:
```
mkdir /ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector_sample/predictions/

TF_RECORD_FILES=$(ls /ai4efs/tfrecords/caltechcameratraps/oneclass/eccv_18/train-?????-of-????? | tr '\n' ',')

python tfmodels/research/object_detection/inference/infer_detections.py \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector_sample/predictions/ss_test_detections_imerit_batch_3.tfrecord-00000-of-00001 \
--inference_graph=/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector_sample/frozen_inference_graph.pb \
--discard_image_pixels
```
Then you can use tfrecords/read_from_tfrecords.py to read detection results from inference tfrecords into python dicts


# Evaluation
detection_eval contains code for evaluating models, based on the python dicts returned from tfrecords/read_from_tfrecords and/or the bash scripts for inference.

Evaluation scripts provided include evaluating at object, image, and sequence levels, evaluating detection models as classifiers (which allows you to evaluate on data that has only class-level annotations), evaluating models per-camera-location, and evaluating models per-species





