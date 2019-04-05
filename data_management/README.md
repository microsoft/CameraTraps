# Overview

Everything in this directory creates or operates on COCO Camera Traps databases, which are .json files structured as...

```
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation]
}

info{
  "year" : int,
  "version" : str,
  "description" : str,
  "contributor" : str
  "date_created" : datetime
}

image{
  "id" : str,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "rights_holder" : str,
  "location": int,
  "datetime": datetime,
  "seq_id": str,
  "seq_num_frames": int,
  "frame_num": int
}

category{
  "id" : int,
  "name" : str
}

annotation{
  "id" : str,
  "image_id" : str,
  "category_id" : int,
  # These are in absolute, floating-point coordinates, with the origin at the upper-left
  "bbox": [x,y,width,height]
}
```

`seq_num_frames` is the total number of frames in the sequence that this image belongs to.

`frame_num` specifies this frame's order in the sequence.

Additional fields may be present for specific data sets.


# Contents

This directory is organized into the following subdirectories...

## root

Miscellaneous tools for manipulating COCO Camera Traps .json files.  Of particular note is `sanity_check_json_db.py`, which validates that a CCT database is well-formatted, optionally checking image existence and size.


## annotations

Code for creating new bounding box annotation tasks and converting annotations to COCO Camera Traps format.


## importers

Code for converting frequently-used metadata formats (or sometimes one-off data sets) to COCO Camera Traps .json files.


## tfrecords

Code for generating tfrecords from COCO Camera Traps .json files.  This directory is based on the [Visipedia tfrecords repo](https://github.com/visipedia/tfrecords).
