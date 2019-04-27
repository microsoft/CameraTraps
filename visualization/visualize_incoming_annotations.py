#####
#
# visualize_incoming_annotations.py
#
# Spot-check the annotations received from iMerit by visualizing annotated bounding 
# boxes on a sample of images and display them in HTML.
#
#####

#%% Imports

import json
import os
import re
import pandas as pd
from tqdm import tqdm

import visualization_utils as vis_utils

# Assumes ai4eutils is on the path (github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list


#%% Settings - change everything in this section to match your task

num_to_visualize = None  # None if visualize all images

viz_size = (675, 450)   # width by height, in pixels

pandas_random_seed = None  # seed for sampling images from all annotation entries

incoming_annotation_path = './temp/batch8a_IDFG.json'
output_dir = '/home/yasiyu/yasiyu_temp/201904_iMerit_verification/batch8_IDFG_group'
os.makedirs(os.path.join(output_dir, 'rendered_images'), exist_ok=True)

images_dir = '/datadrive/IDFG/IDFG_20190104_images_to_annotate'
# '/datadrive/SS_annotated/imerit_batch7_snapshotserengeti_2018_10_26/images'
# '/home/yasiyu/mnt/wildlifeblobssc/rspb/gola/gola_camtrapr_data'
# '/datadrive/IDFG/IDFG_20190104_images_to_annotate'
# '/datadrive/emammal'

# functions for translating from image_id in the annotation files to path to images in images_dir
def default_image_id_to_path(image_id, images_dir):
    return os.path.join(images_dir, image_id)

def emammal_image_id_to_path(image_id, images_dir):
    # the dash between seq and frame is different among the batches
    pattern = re.compile('^datasetemammal\.project(.+?)\.deployment(.+?)\.seq(.+?)[-_]frame(.+?)\.img(.+?)\.')
    match = pattern.match(image_id)
    project_id, deployment_id, seq_id, frame_order, image_id = match.group(1, 2, 3, 4, 5)
    img_path1 = os.path.join(images_dir, '{}{}/{}.jpg'.format(project_id, deployment_id, image_id))
    img_path2 = os.path.join(images_dir, '{}{}/{}.JPG'.format(project_id, deployment_id, image_id))
    img_path = img_path1 if os.path.exists(img_path1) else img_path2
    return img_path

def rspb_image_id_to_path(image_id, images_dir):
    parts = image_id.split('__')
    return os.path.join(images_dir, parts[0], parts[1], image_id)

def ss_batch5_image_id_to_path(image_id, images_dir):
    return os.path.join(images_dir, image_id.replace('-frame', '.frame'))

# specify which of the above functions to use for your dataset
image_id_to_path_func = ss_batch5_image_id_to_path


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


#%% Get a numerical to English label map; note that both the numerical key and the name are str

label_map = {}
for cat in entry['categories']:
    label_map[int(cat['id'])] = cat['name']


#%% Visualize the bboxes on a sample of images
    
if num_to_visualize is not None:
    df_img = df_img.sample(n=num_to_visualize, random_state=pandas_random_seed)

images_html = []
for i in tqdm(range(len(df_img))):
    img_name = df_img.iloc[i]['file_name']
    img_path = image_id_to_path_func(img_name, images_dir)

    if not os.path.exists(img_path):
        print('Image {} cannot be found at the path.'.format(img_path))
        continue

    annos_i = df_anno.loc[df_anno['image_id'] == img_name, :]  # all annotations on this image

    # if len(annos_i) < 20:
    #     continue

    # if len(images_html) > 400:  # cap on maximum to save
    #     break

    try:
        image = vis_utils.open_image(img_path).resize(viz_size)
    except Exception as e:
        print('Image {} failed to open. Error: {}'.format(img_path, e))
        continue

    # only save images with a particular class
    # classes = list(annos_i.loc[:, 'category_id'])
    # classes = [str(i) for i in classes]
    # if '3' not in classes:  # only save images with the 'group' class
    #     continue

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
