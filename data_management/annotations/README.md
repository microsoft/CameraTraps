# Image annotations

This directory contains scripts for creating new bounding box annotation tasks and adding annotations to the database.

## Creating new bounding box annotation tasks

We send all images that need to have their bounding boxes annotated in a flat folder zipped together.

A database in the MegaDB format should be created with any known labels and information prior to sending the images to the annotators. 

The images to be annotated should be named:
```
dataset<dataset>.seq<seq_id>.frame<frame_num>.jpg
```

The annotators will only use the `seq<seq_id>` field to group sequences together. 

If the dataset has no explicit sequence information, it might be best to have the `seq_id` set to a short version of the image name (with `dummy_` prepended) so if the sequence information is in the file names, the sequence can appear together.
