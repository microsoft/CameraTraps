#
# annotation_constants.py
#
# Shared constants used to interpret annotation output
#

# Categories assigned to bounding boxes.  Used throughout our repo; do not change unless
# you are Dan or Siyu.  In fact, do not change unless you are both Dan *and* Siyu.
#
# We use integer indices here; this is different than the API output .json file, 
# where indices are string integers.
bbox_categories = [
    {'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'animal'},
    {'id': 2, 'name': 'person'},
    {'id': 3, 'name': 'group'},  # group of animals
    {'id': 4, 'name': 'vehicle'}
]

bbox_category_id_to_name = {}
bbox_category_name_to_id = {}

for cat in bbox_categories:
    bbox_category_id_to_name[cat['id']] = cat['name']
    bbox_category_name_to_id[cat['name']] = cat['id']
