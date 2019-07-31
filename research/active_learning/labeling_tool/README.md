# Labeling Tool for Species Classification with Active Learning
> Web interface for labeling species in camera trap images

This web interface allows users to annotate cropped images for training a species classification model with active learning. Built with [`bottle`](https://bottlepy.org/docs/dev/).

## Setup
1. Create an `images` folder containing full-size camera trap images, which may be in nested directories.
2. (Required if COCO bounding-box annotations are not available, otherwise optional.) Use `run_tf_detector_batch.py` to get predicted bounding boxes in an `detector_output.csv` file.
3. Create a `crops` folder containing cropped images either obtained from COCO bounding-box annotations (from running `crop_images_from_coco_bboxes.py` in `data_preprocessing`) or from bounding boxes predicted by a detector (`crop_images_from_batch_api_detections.py`). These scripts also create a `crops.json` file in the `crops` folder that contains information about each cropped image and its source file.
4. Create a text file listing the classes to use while labeling the species.
5. Initialize a PostgreSQL database and populate it using `initialize_target_db.py` in `Database`.

## Usage
```bash
python runapp.py --db_name MYDATABASE --db_user USERNAME --db_password PASSWORD --crop_dir PATH_TO_CROPS --class_list class_lists/MYCLASSLIST.TXT --embedding_checkpoint PATH_TO_EMBEDDING_MODEL --checkpoint_dir PATH_TO_OUTPUT_CHECKPOINT_DIR
```

[WebUI](https://github.com/microsoft/CameraTraps/blob/amrita/research/active_learning/labeling_tool/labeling_tool.png)