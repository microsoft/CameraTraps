# Active Learning

This directory constitutes an ongoing project to employ active learning for semi-automated camera trap (and possibly aerial/drone) image labeling.

## Species Classification
We suppose we have a large species classification dataset with labeled images. We run these images through a detector to get crops of the detections (we call this the *pre-training set*), and then learn an _embedding model_ on the labeled crops. We use this embedding model as a feature extractor to build a classifier on a new target dataset.

### Preparing the Pretraining Set
1. Use `CameraTraps/data_management/importers/eMammal/copy_and_unzip_emammal.py` to download and unzip data from Azure storage account to, e.g. a mounted `/datadrive`. This creates a bunch of folders each containing `.JPG` files and a `deployment_manifest.xml` file.



Suppose we have a large dataset of labeled images (e.g. eMammal dataset), and have run these images through a detector to get crops of the detections using `CameraTraps/data_management/databases/classification/make_classification_dataset.py`. We first learn an _embedding model_ on the labeled dataset, which we will use as a feature extractor to build a classifier for our target dataset. This can be done by calling `main.py`.

We also prepare crops for images from our target dataset (e.g. Snapshot Serengeti). We pass each crop through the embedding to obtain a feature vector, ask the oracle (e.g. human expert) to label ~1000 images, and then train a _classification model_ using these labeled samples.

1. `/archive/parallel_run_tf_detector.py`

First, a detector is used on a set of images to get object detection bounding boxes and confidence scores, as well as to prepare crops of the images. Then, a `.csv` file called `detections_XYZ.csv` is generated with the following fields:

| Image Name | Confidence Score | BBox_X! | BBox_Y1 | BBox_X2 | BBox_Y2 |
|-----------:|:-----------------|--------:|--------:|--------:|---------|
| string     | float            |float    |float    |float    |float    |

2. `init.py`

Then, an SQL database is initialized from the output `.csv` file. This database has the following tables:
    
   1. `category`

   |   id   | name | abbr |
   |:------:|:----:|:----:|
   | int    | str  | str  |

   2. `image`
   
   | id |file_name|width|height|location|datetime|frame_num|seq_id|seq_num_frames|
   |:--:|:-------:|:---:|:----:|:------:|:------:|:-------:|:----:|:------------:|
   | str| str     | int | int  | str    |datetime| int     | str  | int          |

   3. `detection`

   | id |image_id|kind|category_id|category_confidence|bbox_confidence|bbox_X1|bbox_Y1|bbox_X2|bbox_Y2|
   |:--:|:------:|:--:|:---------:|:-----------------:|:-------------:|:-----:|:-----:|:-----:|:-----:|
   |str | str    | int| int       | float             |float          | float | float | float | float |

   4. `oracle`

   |   id   |detection_id|label|
   |:------:|:----------:|:---:|
   | int    | str        | int |

3. `main.py`

This trains an embedding model on some crops.

### DL
