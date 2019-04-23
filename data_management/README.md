# Overview

Everything in this directory creates or operates on COCO Camera Traps databases, which are .json files structured as...

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
  "width" : int,
  "height" : int,
  "file_name" : str,
  
  # Optional
  "rights_holder" : str,
  "location": str or int,
  "datetime": datetime,  
  "seq_id": str,
  "seq_num_frames": int,
  "frame_num": int
}

category
{
  # Required
  
  # Category ID 0 generally reserved for the class "empty"  
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

Fields listed as "optional" are intended to standardize commonly-used parameters (such as date/time information).  When present, fields should follow the above conventions.  Additional fields may be present for specific data sets.

Whenever possible, the category ID 0 is associated with a class called "empty", even if there are no empty images in a data set.  When preparing data sets, we normalize all versions of "empty" (such as "none", "Empty", "no animal", etc.) to "empty".

# Contents

This directory is organized into the following subdirectories...


## databases

Miscellaneous tools for manipulating COCO Camera Traps .json files.  Of particular note is `sanity_check_json_db.py`, which validates that a CCT database is well-formatted, optionally checking image existence and size.


## annotations

Code for creating new bounding box annotation tasks and converting annotations to COCO Camera Traps format.


## importers

Code for converting frequently-used metadata formats (or sometimes one-off data sets) to COCO Camera Traps .json files.


## tfrecords

Code for generating tfrecords from COCO Camera Traps .json files.  This directory is based on the [Visipedia tfrecords repo](https://github.com/visipedia/tfrecords).


## classification_dataset

Scripts for creating and analyzing a dataset for classification specifically.
