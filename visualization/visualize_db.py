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
import time
from itertools import compress
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm
import humanfriendly

# Assumes ai4eutils is on the path (github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list

# Assumes the cameratraps repo root is on the path
import visualization.visualization_utils as vis_utils
from data_management.cct_json_utils import IndexedJsonDb


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
    include_filename_links = False
    
    box_thickness = 4
    box_expansion = 0
    
    # These are mutually exclusive; both are category names, not IDs
    classes_to_exclude = None
    classes_to_include = None
    
    # Special tag used to say "show me all images with multiple categories"
    multiple_categories_tag = '*multiple*'

    # We sometimes flatten image directories by replacing a path separator with 
    # another character.  Leave blank for the typical case where this isn't necessary.
    pathsep_replacement = '' # '~'

    # Control rendering parallelization
    parallelize_rendering_n_cores = 25
    
    # Process-based parallelization in this function is currently unsupported
    # due to pickling issues I didn't care to look at, but I'm going to just
    # flip this with a warning, since I intend to support it in the future.
    parallelize_rendering_with_threads = True
    parallelize_rendering = False
    

#%% Helper functions

# Translate the file name in an image entry in the json database to a path, possibly doing
# some manipulation of path separators
def image_filename_to_path(image_file_name, image_base_dir, pathsep_replacement=''):
    
    if len(pathsep_replacement) > 0:
        image_file_name = os.path.normpath(image_file_name).replace(os.pathsep,pathsep_replacement)        
    return os.path.join(image_base_dir, image_file_name)


#%% Core functions

def process_images(db_path, output_dir, image_base_dir, options=None):
    """
    Writes images and html to output_dir to visualize the annotations in the json file
    db_path.
    
    db_path can also be a previously-loaded database.
    
    Returns the html filename and the database:
        
    return htmlOutputFile,image_db
    """    
    
    if options is None:
        options = DbVizOptions()
    
    if not options.parallelize_rendering_with_threads:
        print('Warning: process-based parallelization is not yet supported by visualize_db')
        options.parallelize_rendering_with_threads = True
        
    print(options.__dict__)
    
    if image_base_dir.startswith('http'):
        if not image_base_dir.endswith('/'):
            image_base_dir += '/'
    else:
        assert(os.path.isdir(image_base_dir))
            
    os.makedirs(os.path.join(output_dir, 'rendered_images'), exist_ok=True)
    
    if isinstance(db_path,str):
        assert(os.path.isfile(db_path))    
        print('Loading database from {}...'.format(db_path))
        image_db = json.load(open(db_path))
        print('...done')
    elif isinstance(db_path,dict):
        print('Using previously-loaded DB')
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
                
    # Optionally include/remove images with specific labels, *before* sampling
        
    assert (not ((options.classes_to_exclude is not None) and \
                 (options.classes_to_include is not None))), \
                 'Cannot specify an inclusion and exclusion list'
        
    if (options.classes_to_exclude is not None) or (options.classes_to_include is not None):
     
        print('Indexing database')
        indexed_db = IndexedJsonDb(image_db)
        bValidClass = [True] * len(images)        
        for iImage,image in enumerate(images):
            classes = indexed_db.get_classes_for_image(image)
            if options.classes_to_exclude is not None:
                for excludedClass in options.classes_to_exclude:
                    if excludedClass in classes:
                       bValidClass[iImage] = False
                       break
            elif options.classes_to_include is not None:
                bValidClass[iImage] = False
                if options.multiple_categories_tag in options.classes_to_include:
                    if len(classes) > 1:
                        bValidClass[iImage] = True        
                if not bValidClass[iImage]:
                    for c in classes:
                        if c in options.classes_to_include:
                            bValidClass[iImage] = True
                            break                        
            else:
                raise ValueError('Illegal include/exclude combination')
                
        imagesWithValidClasses = list(compress(images, bValidClass))
        images = imagesWithValidClasses    
    
    
    # Put the annotations in a dataframe so we can select all annotations for a given image
    print('Creating data frames')
    df_anno = pd.DataFrame(annotations)
    df_img = pd.DataFrame(images)
    
    # Construct label map
    label_map = {}
    for cat in categories:
        label_map[int(cat['id'])] = cat['name']
    
    # Take a sample of images
    if options.num_to_visualize is not None:
        df_img = df_img.sample(n=options.num_to_visualize,random_state=options.random_seed)
    
    images_html = []
    
    # Set of dicts representing inputs to render_db_bounding_boxes:
    #
    # bboxes, boxClasses, image_path
    rendering_info = []
    
    print('Preparing rendering list')
    
    for iImage,img in tqdm(df_img.iterrows(),total=len(df_img)):
        
        img_id = img['id']
        assert img_id is not None
        
        img_relative_path = img['file_name']
        
        if image_base_dir.startswith('http'):
            img_path = image_base_dir + img_relative_path
        else:
            img_path = os.path.join(image_base_dir, 
                                    image_filename_to_path(img_relative_path, image_base_dir))
    
        annos_i = df_anno.loc[df_anno['image_id'] == img_id, :] # all annotations on this image
    
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
                categoryName = '<a href="https://www.bing.com/images/search?q={}">{}</a>'.format(
                    categoryName,categoryName)
            imageCategories.add(categoryName)
            
            if 'bbox' in anno:
                bbox = anno['bbox']        
                if isinstance(bbox,float):
                    assert math.isnan(bbox), "I shouldn't see a bbox that's neither a box nor NaN"
                    continue
                bboxes.append(bbox)
                boxClasses.append(anno['category_id'])
        
        # ...for each of this image's annotations
        
        imageClasses = ', '.join(imageCategories)
                
        img_id_string = str(img_id).lower()        
        file_name = '{}_gt.jpg'.format(img_id_string.split('.jpg')[0])
        file_name = file_name.replace('/', '~').replace('\\','~').replace(':','~')
        
        rendering_info.append({'bboxes':bboxes, 'boxClasses':boxClasses, 'img_path':img_path,
                               'output_file_name':file_name})
                
        labelLevelString = ' '
        if len(annotationLevelForImage) > 0:
            labelLevelString = ' (annotation level: {})'.format(annotationLevelForImage)
            
        if 'frame_num' in img and 'seq_num_frames' in img:
            frameString = ' frame: {} of {}, '.format(img['frame_num'],img['seq_num_frames'])
        elif 'frame_num' in img:
            frameString = ' frame: {}, '.format(img['frame_num'])
        else:
            frameString = ' '
        
        filename_text = img_relative_path
        if options.include_filename_links:
            filename_text = '<a href="{}">{}</a>'.format(img_path,img_relative_path)
            
        # We're adding html for an image before we render it, so it's possible this image will
        # fail to render.  For applications where this script is being used to debua a database
        # (the common case?), this is useful behavior, for other applications, this is annoying.
        images_html.append({
            'filename': '{}/{}'.format('rendered_images', file_name),
            'title': '{}<br/>{}, num boxes: {}, {}class labels: {}{}'.format(
                filename_text, img_id, len(bboxes), frameString, imageClasses, labelLevelString),
            'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;' + \
                'text-align:left;margin-top:20;margin-bottom:5'
        })
    
    # ...for each image

    def render_image_info(rendering_info):
        
        img_path = rendering_info['img_path']
        bboxes = rendering_info['bboxes']
        bboxClasses = rendering_info['boxClasses']
        output_file_name = rendering_info['output_file_name']
        
        if not img_path.startswith('http'):
            if not os.path.exists(img_path):
                print('Image {} cannot be found'.format(img_path))
                return False
            
        try:
            original_image = vis_utils.open_image(img_path)
            original_size = original_image.size
            if options.viz_size[0] == -1 and options.viz_size[1] == -1:
                image = original_image
            else:
                image = vis_utils.resize_image(original_image, options.viz_size[0],
                                               options.viz_size[1])
        except Exception as e:
            print('Image {} failed to open. Error: {}'.format(img_path, e))
            return False
            
        vis_utils.render_db_bounding_boxes(boxes=bboxes, classes=bboxClasses,
                                           image=image, original_size=original_size,
                                           label_map=label_map,
                                           thickness=options.box_thickness,
                                           expansion=options.box_expansion)
        
        image.save(os.path.join(output_dir, 'rendered_images', output_file_name))
        return True
    
    # ...def render_image_info
    
    print('Rendering images')
    start_time = time.time()
    
    if options.parallelize_rendering:
        
        if options.parallelize_rendering_with_threads:
            worker_string = 'threads'
        else:
            worker_string = 'processes'
            
        if options.parallelize_rendering_n_cores is None:
            if options.parallelize_rendering_with_threads:
                pool = ThreadPool()
            else:
                pool = Pool()
        else:
            if options.parallelize_rendering_with_threads:
                pool = ThreadPool(options.parallelize_rendering_n_cores)
            else:
                pool = Pool(options.parallelize_rendering_n_cores)
            print('Rendering images with {} {}'.format(options.parallelize_rendering_n_cores,
                                                       worker_string))            
        rendering_success = list(tqdm(pool.imap(render_image_info, rendering_info),
                                 total=len(rendering_info)))
        
    else:
        
        rendering_success = []
        for file_info in tqdm(rendering_info):        
            rendering_success.append(render_image_info(file_info))
            
    elapsed = time.time() - start_time
    
    print('Rendered {} images in {} ({} successful)'.format(
        len(rendering_info),humanfriendly.format_timespan(elapsed),sum(rendering_success)))
        
    if options.sort_by_filename:    
        images_html = sorted(images_html, key=lambda x: x['filename'])
        
    htmlOutputFile = os.path.join(output_dir, 'index.html')
    
    htmlOptions = options.htmlOptions
    if isinstance(db_path,str):
        htmlOptions['headerHtml'] = '<h1>Sample annotations from {}</h1>'.format(db_path)
    else:
        htmlOptions['headerHtml'] = '<h1>Sample annotations</h1>'
        
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
                        help='Base directory (or URL) for input images')

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
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
            
    args = parser.parse_args()
    
    # Convert to an options object
    options = DbVizOptions()
    args_to_object(args, options)
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
    
    htmlOutputFile,db = process_images(db_path,output_dir,image_base_dir,options)
    # os.startfile(htmlOutputFile)

