#
# Spot check the annotations received from iMerit by visualizing annotated bounding boxes on a sample of images
# and display them in an HTML.
#

#%% Imports
import json
import os
import re

import pandas as pd
from tqdm import tqdm

import visualization_utils as vis_utils

import sys
sys.path.append('/home/yasiyu/repos/ai4eutils')  # path to the https://github.com/Microsoft/ai4eutils repo
from write_html_image_list import write_html_image_list

#%% Settings - change everything in this section to match your task
num_to_visualize = 500

viz_size = (675, 450)   # width by height, in pixels

incoming_annotation_path = 'iMerit_annotations.json'
output_dir = ''

images_dir = ''

os.makedirs(os.path.join(output_dir, 'rendered_images'), exist_ok=True)

# functions for translating from image_id in the annotation files to path to images in images_dir
def emammal_image_id_to_path(image_id, images_dir):
    # the dash between seq and frame is different among the batches
    pattern = re.compile('^datasetemammal\.project(.+?)\.deployment(.+?)\.seq(.+?)[-_]frame(.+?)\.img(.+?)\.')
    match = pattern.match(image_id)
    project_id, deployment_id, seq_id, frame_order, image_id = match.group(1, 2, 3, 4, 5)
    img_path1 = os.path.join(images_dir, '{}{}/{}.jpg'.format(project_id, deployment_id, image_id))
    img_path2 = os.path.join(images_dir, '{}{}/{}.JPG'.format(project_id, deployment_id, image_id))
    img_path = img_path1 if os.path.exists(img_path1) else img_path2
    return img_path

def idfg_image_id_to_path(image_id, images_dir):
    return os.path.join(images_dir, image_id)

def rspb_image_id_to_path(image_id, images_dir):
    parts = image_id.split('__')
    return os.path.join(images_dir, parts[0], parts[1], image_id)

image_id_to_path_func = idfg_image_id_to_path


#%% Read in the annotations
with open(incoming_annotation_path, 'r') as f:
    content = f.readlines()

print('Incoming annotations at {} has {} rows.'.format(incoming_annotation_path, len(content)))

# put the annotations in a dataframes so we can select all annotations for a given image
annotations = []
images = []
for row in content:
    entry = json.loads(row)
    annotations.extend(entry['annotations'])
    images.extend(entry['images'])

df_anno = pd.DataFrame(annotations)
df_img = pd.DataFrame(images)


#%% Get a numerical to English label map
label_map = {}
for cat in entry['categories']:
    label_map[cat['id']] = cat['name']


#%% Visualize the bboxes on a sample of images
sample_img = df_img.sample(n=num_to_visualize)

images_html = []
for i in tqdm(range(len(sample_img))):
    img_name = sample_img.iloc[i]['file_name']
    img_path = image_id_to_path_func(img_name, images_dir)

    if not os.path.exists(img_path):
        print('Image {} cannot be found at the path.'.format(img_path))
        continue

    annos_i = df_anno.loc[df_anno['image_id'] == img_name, :]  # all annotations on this image

    try:
        image = vis_utils.open_image(img_path).resize(viz_size)
    except Exception as e:
        print('Image {} failed to open. Error: {}'.format(img_path, e))
        continue

    if len(annos_i) > 0:
        bboxes = list(annos_i.loc[:, 'bbox'])
        classes = list(annos_i.loc[:, 'category_id'])
        vis_utils.render_iMerit_boxes(bboxes, classes, image, label_map)  # image changed in place

    file_name = '{}_gtbbox.jpg'.format(img_name.lower().split('.jpg')[0])
    image.save(os.path.join(output_dir, 'rendered_images', file_name))

    images_html.append({
        'filename': '{}/{}'.format('rendered_images', file_name),
        'title': '{}, number of boxes: {}'.format(img_name, len(annos_i)),
        'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
    })


#%% Write to HTML
images_html = sorted(images_html, key=lambda x: x['filename'])
write_html_image_list(
        filename=os.path.join(output_dir, 'index.html'),
        images=images_html,
        options={
            'headerHtml': '<h1>Sample annotations from {}</h1>'.format(incoming_annotation_path)
        })

print('Visualized {} images.'.format(len(images_html)))
