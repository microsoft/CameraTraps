# MegaDB

Internally we store all labels and metadata associated with each image sequence in a NoSQL database for easy querying.

The database allows for any properties to be associated at the sequence, image and bounding box levels.

We do not enforce a taxonomy for animal class / species labels: the original labels provided by the organization are kept, with minor typo and consistency corrections.

It allows for queries such as 
- get all the publicly released images with the class label "bear" in sequences of length greater than 2
- get all images in the dataset "caltech" with one or more bounding boxes
- get all species in the database

All images are stored in blob storage, so you can download all the images returned by a query.


## Structure

There are two tables in this database. 
- `sequences`
Each item in this list is a `sequence` object. 

- `datasets`
This table stores information about each dataset in the database. Each dataset object contains information on where in blob storage its images are stored, including any path prefix, its public access status and any other dataset-level properties.



## Conventions

While most of the enforceable rules are included in the schema `schema.json` and the extra checks in `verify_items_to_insert.py`, we observe a number of additional conventions detailed here. Some rules enforced by the schema and additional checks are explained here also for clarity.

### Which level to associate a property
We always associated a property with the highest level in the hierachy that is still correct. 
- If the `rights_holder` is the same across a dataset, this property should be on the dataset object in the `datasets` table.
- If `class`, usually used for species or other fine animal categories, is labeled on sequences, this property should be at the `sequence` level, not the `image` level. 


### Sequence information
- For images whose sequence information is unknown, each image is contained in a sequence object whose `seq_id` will start with `dummy_`.
- The `frame_num` property on each image object is kept to be safe, even though it is redundant because the image objects are in a list. Actually, image items in the list are not ordered according to `frame_num`. `frame_num` need to be unique, but does not need to be consecutive. The min value for `frame_num` is 0 even though most start at 1.
- The `location` property can be a string or an object (e.g. latitude/longitude).


### When labels are unavailable
- The database we have is intended for *labeled* images, but in some datasets it makes sense to keep subsets of images that do not have labels. For those, their `class` property will be a list of only the label "__label_unavailable". This is different from "unidentified", which may be an animal that does not have the finer category label.
- Since we have this special keyword for unavailable labels, each sequence OR all images in a sequence should have the `class` property.


### Images
- Images in a sequence are stored in the `images` field.
- We do not store the height and width of the images 
    - when creating training data the images will need to be opened and their sizes read then
    - all bounding box and other coordinate annotations are stored in relative coordinates.
    
#### Empty images
- The one `class` where we do enforce a "taxonomy": all images that are labeled as emtpy of objects of interest should have its `class` property be a list with only the "empty" label. Do not use "blank", "nothing", etc. 


### Bounding boxes labels for images
- Bounding boxes are stored in the `bbox` field in an image object. The coordinates in the `bbox` sub-field is `[x_min, y_min, w, h]`, the top-left corner of the box and its width and height, all relative to the width and height of the image.
- If an image was sent for annotation but was determined to be empty, the image entry will still have a `bbox` field that points to an empty list. We save this information as a more reliable "empty" label.


## Future

### Videos
- When we have video data, it will be added as a property of the sequence object at the same level as `images`.

### More properties associated with each class
- We will create another table `classes` that will associate a `class` property value with additional fields, for each dataset.
