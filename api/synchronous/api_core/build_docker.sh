#!/bin/bash

# These dependency files are outside of the Docker context, so we cannot 
# use the COPY action in the Dockerfile to copy them into the Docker image.

# This is the main dependency
mkdir animal_detection_api/detection/
cp -a ../../../detection/. animal_detection_api/detection/

# Copy YOLOv5 dependencies
git clone https://github.com/ultralytics/yolov5/
cd yolov5

# To ensure forward-compatibility, we pin a particular version
# of YOLOv5
git checkout c23a441c9df7ca9b1f275e8c8719c949269160d1
cd ../

# Copy other required dependencies from our repo 
cp ../../../ct_utils.py animal_detection_api/

mkdir -p animal_detection_api/visualization/
cp ../../../visualization/visualization_utils.py animal_detection_api/visualization/

mkdir -p animal_detection_api/data_management
mkdir -p animal_detection_api/data_management/annotations/
cp ../../../data_management/annotations/annotation_constants.py animal_detection_api/data_management/annotations/

echo $1
echo $2
sudo docker build . --build-arg BASE_IMAGE=$1 -t $2 

