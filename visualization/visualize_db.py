########
#
# visualize_db.py
# 
# Outputs an HTML page visualizing annotations (class labels and/or bounding boxes)
# on a sample of images in a database in the COCO Camera Traps format
#
########

#%% Imports

import argparse
import inspect
import json
import math
import os
import sys
from itertools import compress

import pandas as pd
from tqdm import tqdm

# Assumes ai4eutils is on the path (github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list

# Assumes the cameratraps repo root is on the path
import visualization.visualization_utils as vis_utils


#%% Settings

class DbVizOptions:
    
    # Set to None to visualize all images
    num_to_visualize = None
    
    # Target size for rendering; set either dimension to -1 to preserve aspect ratio
    viz_size = (675, -1)
    htmlOptions = write_html_image_list()
    sort_by_filename = True
    trim_to_images_with_bboxes = False
    random_seed = 0 # None
    add_search_links = False

    # We sometimes flatten image directories by replacing a path separator with 
    # another character.  Leave blank for the typical case where this isn't necessary.
    pathsep_replacement = '' # '~'


#%% Helper functions

# Translate the file name in an image entry in the json database to a path, possibly doing
# some manipulation of path separators
def image_filename_to_path(image_file_name, image_base_dir, pathsep_replacement=''):
    
    if len(pathsep_replacement) > 0:
        image_file_name = os.path.normpath(image_file_name).replace(os.pathsep,pathsep_replacement)        
    return os.path.join(image_base_dir, image_file_name)


#%% Core functions

def process_images(db_path,output_dir,image_base_dir,options=None):
    """
    Writes images and html to output_dir to visualize the annotations in the json file
    db_path.
    
    db_path can also be a previously-loaded database.
    
    Returns the html filename and the database:
        
    return htmlOutputFile,image_db
    """    
    
    if options is None:
        options = DbVizOptions()
    
    print(options.__dict__)
    
    os.makedirs(os.path.join(output_dir, 'rendered_images'), exist_ok=True)
    assert(os.path.isdir(image_base_dir))
    
    if isinstance(db_path,str):
        assert(os.path.isfile(db_path))    
        print('Loading database from {}...'.format(db_path))
        image_db = json.load(open(db_path))
        print('...done')
    elif isinstance(db_path,dict):
        image_db = db_path
    else:
        raise ValueError('Illegal dictionary or filename')
        
    annotations = image_db['annotations']
    images = image_db['images']
    categories = image_db['categories']
    
    # Optionally remove all images without bounding boxes, *before* sampling
    if options.trim_to_images_with_bboxes:
        
        bHasBbox = [False] * len(annotations)
        for iAnn,ann in enumerate(annotations):
            if 'bbox' in ann:
                assert isinstance(ann['bbox'],list)
                bHasBbox[iAnn] = True
        annotationsWithBboxes = list(compress(annotations, bHasBbox))
        
        imageIDsWithBboxes = [x['image_id'] for x in annotationsWithBboxes]
        imageIDsWithBboxes = set(imageIDsWithBboxes)
        
        bImageHasBbox = [False] * len(images)
        for iImage,image in enumerate(images):
            imageID = image['id']
            if imageID in imageIDsWithBboxes:
                bImageHasBbox[iImage] = True
        imagesWithBboxes = list(compress(images, bImageHasBbox))
        images = imagesWithBboxes
        
    # put the annotations in a dataframe so we can select all annotations for a given image
    df_anno = pd.DataFrame(annotations)
    df_img = pd.DataFrame(images)
    
    # construct label map
    label_map = {}
    for cat in categories:
        label_map[int(cat['id'])] = cat['name']
    
    # take a sample of images
    if options.num_to_visualize is not None:
        df_img = df_img.sample(n=options.num_to_visualize,random_state=options.random_seed)
    
    images_html = []
    
    # iImage = 0
    for iImage in tqdm(range(len(df_img))):
        
        img_id = df_img.iloc[iImage]['id']
        img_relative_path = df_img.iloc[iImage]['file_name']
        img_path = os.path.join(image_base_dir, image_filename_to_path(img_relative_path, image_base_dir))
    
        if not os.path.exists(img_path):
            print('Image {} cannot be found'.format(img_path))
            continue
    
        annos_i = df_anno.loc[df_anno['image_id'] == img_id, :]  # all annotations on this image
    
        try:
            originalImage = vis_utils.open_image(img_path)
            original_size = originalImage .size
            image = vis_utils.resize_image(originalImage , options.viz_size[0], options.viz_size[1])
        except Exception as e:
            print('Image {} failed to open. Error: {}'.format(img_path, e))
            continue
    
        bboxes = []
        boxClasses = []
        
        # All the class labels we've seen for this image (with out without bboxes)
        imageCategories = set()
        
        annotationLevelForImage = ''
        
        # Iterate over annotations for this image
        # iAnn = 0; anno = annos_i.iloc[iAnn]
        for iAnn,anno in annos_i.iterrows():
        
            if 'sequence_level_annotation' in anno:
                bSequenceLevelAnnotation = anno['sequence_level_annotation']
                if bSequenceLevelAnnotation:
                    annLevel = 'sequence'
                else:
                    annLevel = 'image'
                if annotationLevelForImage == '':
                    annotationLevelForImage = annLevel
                elif annotationLevelForImage != annLevel:
                    annotationLevelForImage = 'mixed'
                    
            categoryID = anno['category_id']
            categoryName = label_map[categoryID]
            if options.add_search_links:
                categoryName = categoryName.replace('"','')
                categoryName = '<a href="https://www.bing.com/images/search?q={}">{}</a>'.format(categoryName,categoryName)
            imageCategories.add(categoryName)
            
            if 'bbox' in anno:
                bbox = anno['bbox']        
                if isinstance(bbox,float):
                    assert math.isnan(bbox), "I shouldn't see a bbox that's neither a box nor NaN"
                    continue
                bboxes.append(bbox)
                boxClasses.append(anno['category_id'])
        
        imageClasses = ', '.join(imageCategories)
        
        # render bounding boxes in-place
        vis_utils.render_db_bounding_boxes(bboxes, boxClasses, image, original_size, label_map)  
        
        file_name = '{}_gtbbox.jpg'.format(img_id.lower().split('.jpg')[0])
        file_name = file_name.replace('/', '~')
        image.save(os.path.join(output_dir, 'rendered_images', file_name))
    
        labelLevelString = ''
        if len(annotationLevelForImage) > 0:
            labelLevelString = ' (annotation level: {})'.format(annotationLevelForImage)
            
        images_html.append({
            'filename': '{}/{}'.format('rendered_images', file_name),
            'title': '{}<br/>{}, number of boxes: {}, class labels: {}{}'.format(img_relative_path,img_id, len(bboxes), imageClasses, labelLevelString),
            'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
        })
    
    # ...for each image

    if options.sort_by_filename:    
        images_html = sorted(images_html, key=lambda x: x['filename'])
        
    htmlOutputFile = os.path.join(output_dir, 'index.html')
    
    htmlOptions = options.htmlOptions
    htmlOptions['headerHtml'] = '<h1>Sample annotations from {}</h1>'.format(db_path)
    write_html_image_list(
            filename=htmlOutputFile,
            images=images_html,
            options=htmlOptions)

    print('Visualized {} images, wrote results to {}'.format(len(images_html),htmlOutputFile))
    
    return htmlOutputFile,image_db

# def process_images(...)
    
    
#%% Command-line driver
    
# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check existence in the target object.
def args_to_object(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', action='store', type=str, 
                        help='.json file to visualize')
    parser.add_argument('output_dir', action='store', type=str, 
                        help='Output directory for html and rendered images')
    parser.add_argument('image_base_dir', action='store', type=str, 
                        help='Base directory for input images')

    parser.add_argument('--num_to_visualize', action='store', type=int, default=None, 
                        help='Number of images to visualize (randomly drawn) (defaults to all)')
    parser.add_argument('--random_sort', action='store_true', 
                        help='Sort randomly (rather than by filename) in output html')
    parser.add_argument('--trim_to_images_with_bboxes', action='store_true', 
                        help='Only include images with bounding boxes (defaults to false)')
    parser.add_argument('--random_seed', action='store', type=int, default=None,
                        help='Random seed for image selection')
    parser.add_argument('--pathsep_replacement', action='store', type=str, default='',
                        help='Replace path separators in relative filenames with another character (frequently ~)')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
            
    args = parser.parse_args()
    
    # Convert to an options object
    options = DbVizOptions()
    args_to_object(args,options)
    if options.random_sort:
        options.sort_by_filename = False
        
    process_images(options.db_path,options.output_dir,options.image_base_dir,options) 


if __name__ == '__main__':
    
    main()


#%% Interactive driver(s)

if False:
    
    #%%
    
    db_path = r'e:\wildlife_data\missouri_camera_traps\missouri_camera_traps_set1.json'
    output_dir = r'e:\wildlife_data\missouri_camera_traps\preview'
    image_base_dir = r'e:\wildlife_data\missouri_camera_traps'
    
    options = DbVizOptions()
    options.num_to_visualize = 100
    
    htmlOutputFile = process_images(db_path,output_dir,image_base_dir,options)
    # os.startfile(htmlOutputFile)

