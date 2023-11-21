#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/../tfmodels/research:`pwd`/../tfmodels/research/slim

EXPERIMENT=/ai4efs/sample_object_detection_experiment_directory

tensorboard --logdir $EXPERIMENT_DIR --port 6006 &

python ../tfmodels/research/object_detection/model_main.py --pipeline_config_path=$EXPERIMENT_DIR/configs/pipeline.config --model_dir=$EXPERIMENT_DIR --num_train_steps=200000 --num_eval_steps=1500 --alsologtostderr


