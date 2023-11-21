# data_preprocessing

Produces crops either from detector_output.csv produced by 
run_tf_detector_batch.py (via crop_images_from_batch_api_detections.py)
or from bboxes stored in COCO .json file.

Either way, produces a .json file in the format expected by 
Database/initialize_*.py.

# Database

initialize_*.py populates a database (target or pretrain).  DB_models.py
defines the data representation that's built; this is what's used by 
the rest of the code.

add_oracle_to_db.py adds ground truth to a db for offline experiments.

# experiments

One-off scripts and notebooks.

# labeling_tool

See [labeling_tool/README.md](labeling_tool/README.md).

# DL

## Engine.py

ML utility functions like "train()" and "validate()"

## losses.py

Loss functions: focal, center, triplet, contrastive

## networks.py

EmbeddingNet loads pre-trained embedding models

SoftMaxNet is a wrapper for EmbeddingNet adding softmax loss 

ClassificationNet was the hand-created classification network, but was replaced by a network built into scikit learn

## sqlite_data_loader

Loads a data set from an existing SQLite DB

# sampling_methods

Query tools for image selection for active learning.

Mostly third-party code from:

https://github.com/google/active-learning/tree/master/sampling_methods

New stuff:

entropy
confidence
uniform_sampling (modified)
constants.py (modified)

Also miscellaneous fixes related to database assumptions.