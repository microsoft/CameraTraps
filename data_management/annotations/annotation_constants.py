#
# annotation_constants.py
#
# Shared constants used to interpret annotation output
#

# The four categories for bounding boxes - do not change
# Note that the category ID in the API output json file is of type string, not int as here
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
