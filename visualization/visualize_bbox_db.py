#
# Outputs a HTML page visualizing bounding boxes on a sample of images in a bbox database
# in the COCO Camera Trap format (output of data_management/annotations/add_bounding_boxes_to_json.py).
#

#%% Imports
import json
import os

import pandas as pd
from tqdm import tqdm

import visualization_utils as vis_utils

import sys
sys.path.append('/home/yasiyu/repos/ai4eutils')  # path to the https://github.com/Microsoft/ai4eutils repo
from write_html_image_list import write_html_image_list


#%% Settings
num_to_visualize = 25
viz_size = (675, 450)

bbox_db_path = 'temp/idfg_bboxes_20190409.json'
output_dir = '/home/yasiyu/yasiyu_temp/201904_iMerit_verification/json_bbox_idfg'
os.makedirs(os.path.join(output_dir, 'rendered_images'), exist_ok=True)

# assume that path to the image is the concatenation of images_dir and the file_name field in each image entry
images_dir = '/datadrive/IDFG/IDFG_20190104_images_to_annotate'

# functions for translating from file_name in an image entry in the json database to path to images in images_dir
def default_image_file_name_to_path(image_file_name, images_dir):
    return os.path.join(images_dir, image_file_name)

def idfg_image_file_name_to_path(image_file_name, images_dir):
    return os.path.join(images_dir, image_file_name.replace('/', '~'))

# specify which of the above functions to use for your dataset
image_file_name_to_path_func = idfg_image_file_name_to_path


#%% Processing images
print('Loading the bbox database...')
bbox_db = json.load(open(bbox_db_path))

# put the annotations in a dataframes so we can select all annotations for a given image
df_anno = pd.DataFrame(bbox_db['annotations'])
df_img = pd.DataFrame(bbox_db['images'])

# construct label map
label_map = {}
for cat in bbox_db['categories']:
    label_map[int(cat['id'])] = cat['name']


# take a sample of images
if num_to_visualize is not None:
    df_img = df_img.sample(n=num_to_visualize)

images_html = []
for i in tqdm(range(len(df_img))):
    img_id = df_img.iloc[i]['id']
    img_file_name = df_img.iloc[i]['file_name']
    img_path = os.path.join(images_dir, image_file_name_to_path_func(img_file_name, images_dir))

    if not os.path.exists(img_path):
        print('Image {} cannot be found at the path.'.format(img_path))
        continue

    annos_i = df_anno.loc[df_anno['image_id'] == img_id, :]  # all annotations on this image

    try:
        image = vis_utils.open_image(img_path)
        image_size = image.size
        image = image.resize(viz_size)
    except Exception as e:
        print('Image {} failed to open. Error: {}'.format(img_path, e))
        continue

    if len(annos_i) > 0:
        bboxes = list(annos_i.loc[:, 'bbox'])
        classes = list(annos_i.loc[:, 'category_id'])
        vis_utils.render_db_bounding_boxes(bboxes, classes, image, image_size, label_map)  # image changed in place

    file_name = '{}_gtbbox.jpg'.format(img_id.lower().split('.jpg')[0])
    file_name = file_name.replace('/', '~')
    image.save(os.path.join(output_dir, 'rendered_images', file_name))

    images_html.append({
        'filename': '{}/{}'.format('rendered_images', file_name),
        'title': '{}, number of boxes: {}'.format(img_id, len(annos_i)),
        'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
    })


#%% Write to HTML
images_html = sorted(images_html, key=lambda x: x['filename'])
write_html_image_list(
        filename=os.path.join(output_dir, 'index.html'),
        images=images_html,
        options={
            'headerHtml': '<h1>Sample annotations from {}</h1>'.format(bbox_db_path)
        })

print('Visualized {} images.'.format(len(images_html)))
