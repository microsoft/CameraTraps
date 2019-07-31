# Labeling Tool for Species Classification with Active Learning
> Web interface for labeling species in camera trap images

This web interface allows users to annotate cropped images for training a species classification model with active learning. Built with [`bottle`][https://bottlepy.org/docs/dev/].

## Setup
The app assumes you have a dataset with an `images` folder and a `crops` folder. The `images` folder contains full-size camera trap images, which may be in nested directories. The `crops` folder contains cropped images that are either obtained from COCO bounding-box annotations (from running `crop_images_from_coco_bboxes.py` in `data_preprocessing`) or from bounding boxes predicted by a detector (`crop_images_from_batch_api_detections.py`). Both these scripts also create a `crops.json` file in the `crops` folder that contains information about each cropped image and its source file.