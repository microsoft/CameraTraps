# Announcement

At the core of our mission is the desire to create a harmonious space where conservation scientists from all over the globe can unite, share, and grow. We are expanding the CameraTraps repo to introduce **Pytorch-Wildlife**, a Collaborative Deep Learning Framework for Conservation, where researchers can come together to share and use datasets and deep learning architectures for wildlife conservation.
 
We've been inspired by the potential and capabilities of Megadetector, and we deeply value its contributions to the community. **As we forge ahead with Pytorch-Wildlife, under which Megadetector now resides, please know that we remain committed to supporting, maintaining, and developing Megadetector, ensuring its continued relevance, expansion, and utility.**

To use the newest version of MegaDetector with all the exisitng functionatlities, you can use our newly developed [user interface](#explore-pytorch-wildlife-and-megadetector-with-our-user-interface) or simply load the model with **Pytorch-Wildlife** and the weights will be automatically downloaded:

```python
from PytorchWildlife.models import detection as pw_detection
detection_model = pw_detection.MegaDetectorV5()
```

If you'd like to learn more about **Pytorch-Wildlife**, please continue reading.

For those interested in accessing the previous MegaDetector repository, which utilizes the same `MegaDetector v5` model weights and was primarily developed by Dan Morris during his time at Microsoft, please visit the [archive](./archive) directory, or you can visit this [forked repository](https://github.com/agentmorris/MegaDetector/tree/main) that Dan Morris is actively maintaining.
 
**If you have any questions regarding MegaDetector and Pytorch-Wildlife, please <a href="mailto:zhongqimiao@microsoft.com">email us</a>!**

# Overview

Almost all the scripts in this directory create or operates on COCO Camera Traps databases, which are .json files structured as...

## COCO Camera Traps format

```
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation]
}

info 
{
  # Required
  "version" : str,
  "description" : str,
  
  # Optional
  "year" : int,
  "contributor" : str
  "date_created" : datetime
}

image
{
  # Required
  "id" : str,
  "file_name" : str,
  
  # Optional
  "width" : int,
  "height" : int,
  "rights_holder" : str,    
  "datetime": datetime,  
  "seq_id": str,
  "seq_num_frames": int,
  "frame_num": int
  
  # This is an int in older data sets, but convention is now strings
  "location": str,
  
  # Image corruption is quite common in camera trap images, and throwing out corrupt
  # images in database assembly is "dodging part of the problem".  Wherever possible,
  # use this flag to indicate that an image failed to load, e.g. in PIL and/or TensorFlow.
  "corrupt": bool
}

category
{
  # Required
  
  # Category ID 0 reserved for the class "empty"; all other categories vary by data
  # set.  Non-negative integers only.
  "id" : int,
  "name" : str  
}

annotation
{
  # Required
  "id" : str,
  "image_id" : str,  
  "category_id" : int,
  
  # Optional
  
  # These are in absolute, floating-point coordinates, with the origin at the upper-left
  "bbox": [x,y,width,height],
  
  # This indicates that this annotation is really applied at the *sequence* level,
  # and may not be reliable at the individual-image level.  Since the *sequences* are
  # the "atom of interest" for most ecology applications, this is common.
  "sequence_level_annotation" : bool
}
```

`seq_num_frames` is the total number of frames in the sequence that this image belongs to.

`frame_num` specifies this frame's order in the sequence.

Note that the coordinates in the `bbox` field are absolute here, different from those in the [batch processing API](api/batch_processing/README.md) output, which are relative.

Fields listed as "optional" are intended to standardize commonly-used parameters (such as date/time information).  When present, fields should follow the above conventions.  Additional fields may be present for specific data sets.

Whenever possible, the category ID 0 is associated with a class called "empty", even if there are no empty images in a data set.  When preparing data sets, we normalize all versions of "empty" (such as "none", "Empty", "no animal", etc.) to "empty".

# Contents

This directory is organized into the following subdirectories...

## annotations
Code for creating new bounding box annotation tasks and converting annotations to COCO Camera Traps format.

## databases
Miscellaneous tools for manipulating COCO Camera Traps .json files.  Of particular note is `sanity_check_json_db.py`, which validates that a CCT database is well-formatted, optionally checking image existence and size.

**databases/classifcation**: Scripts for creating and analyzing a dataset for classification specifically.

## importers
Code for converting frequently-used metadata formats (or sometimes one-off data sets) to COCO Camera Traps .json files.

## megadb
Code for querying and updating MegaDB.

## tfrecords
Code for generating tfrecords from COCO Camera Traps .json files.  This directory is based on the [Visipedia tfrecords repo](https://github.com/visipedia/tfrecords).
