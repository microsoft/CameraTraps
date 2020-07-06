#!/bin/bash

# these dependency files are outside of the Docker context, so cannot use the COPY action in the Dockerfile to copy them into the Docker image.

# this is the main dependency
cp ../../../detection/run_tf_detector.py animal_detection_classification_api/

# which depends on the following
cp ../../../ct_utils.py animal_detection_classification_api/

mkdir animal_detection_classification_api/visualization/
cp ../../../visualization/visualization_utils.py animal_detection_classification_api/visualization/

# visualization_utils in turn depends on the following
mkdir animal_detection_classification_api/data_management
mkdir animal_detection_classification_api/data_management/annotations/
cp ../../../data_management/annotations/annotation_constants.py animal_detection_classification_api/data_management/annotations/


# Modify the version and build numbers as needed in the API_DOCKER_IMAGE argument passed to this script
echo $1
sudo docker build . -t $1


