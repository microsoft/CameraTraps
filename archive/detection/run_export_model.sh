#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/../tfmodels/research:`pwd`/../tfmodels/research/slim

EXPERIMENT_DIR=/ai4efs/sample_object_detection_experiment_directory

OUTPUT_MODEL_DIR=/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/

EXPERIMENT=megadetector

CHECKPOINT_NUMBER=132249 #FILL IN WITH YOUR BEST MODEL NUMBER

python ../tfmodels/research/object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=$EXPERIMENT_DIR/$EXPERIMENT/configs/pipeline.config --trained_checkpoint_prefix=$EXPERIMENT_DIR/$EXPERIMENT/model.ckpt-$CHECKPOINT_NUMBER --output_directory=$OUTPUT_MODEL_DIR/$EXPERIMENT/

mkdir $OUTPUT_MODEL_DIR/$EXPERIMENT/predictions/

