"""
compare_batch_results.py

Compare two sets of batch results; typically used to compare MegaDetector versions.

Currently supports only detection results (not classification results).
"""

#%% Imports

import json
import os
import random
import sys
import subprocess

from tqdm import tqdm
from visualization import visualization_utils

# Assumes ai4eutils is on the python path (https://github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list
import path_utils


#%% Constants and support classes

# We will confirm that this matches what we load from each file
detection_categories = {'1': 'animal', '2': 'person', '3': 'vehicle'}
    
def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
    
class BatchComparisonOptions:

    output_folder = None
    image_folder = None
    
    results_filename_a = None
    results_filename_b = None
    
    results_description_a = None
    results_description_b = None
    
    detection_thresholds_a = {'animal':0.7,'person':0.7,'vehicle':0.7}
    detection_thresholds_b = {'animal':0.7,'person':0.7,'vehicle':0.7}

    max_images_per_category = 100
    colormap_a = ['Red']
    colormap_b = ['RoyalBlue']

    rendering_confidence_threshold_a = 0.1
    rendering_confidence_threshold_b = 0.1

    target_width = 800
    
    n_rendering_threads = 50    
    
    random_seed = 0
    
    
class BatchComparisonResults:
    
    html_output_file = None
    
    
#%% Main function

def compare_batch_results(options):
    
    random.seed(options.random_seed)

    ##%% Validate inputs
    
    assert os.path.isfile(options.results_filename_a)
    assert os.path.isfile(options.results_filename_b)
    assert os.path.isdir(options.image_folder)
    os.makedirs(options.output_folder,exist_ok=True)
    
    
    ##%% Load both result sets
    
    with open(options.results_filename_a,'r') as f:
        results_a = json.load(f)
    
    with open(options.results_filename_b,'r') as f:
        results_b = json.load(f)
        
    assert results_a['detection_categories'] == detection_categories
    assert results_b['detection_categories'] == detection_categories
    
    images_a = results_a['images']
    images_b = results_b['images']
    
    filename_to_image_a = {im['file']:im for im in images_a}
    filename_to_image_b = {im['file']:im for im in images_b}
    
    
    ##%% Make sure they represent the same set of images
    
    filenames_a = [im['file'] for im in images_a]
    filenames_b_set = set([im['file'] for im in images_b])
    
    assert len(images_a) == len(images_b)
    assert len(filenames_a) == len(images_a)
    assert len(filenames_b_set) == len(images_b)
    for fn in filenames_a:
        assert fn in filenames_b_set
    
    
    ##%% Find differences
    
    # Each of these maps a filename to a two-element list (the image in set A, the image in set B)
    #
    # Right now, we only handle a very simple notion of class transition, where the detection
    # of maximum confidence changes class *and* both images have an above-threshold detection.
    common_detections = {}
    common_non_detections = {}
    detections_a_only = {}
    detections_b_only = {}
    class_transitions = {}
    
    # fn = filenames_a[0]
    for fn in tqdm(filenames_a):
        
        im_a = filename_to_image_a[fn]
        im_b = filename_to_image_b[fn]
        
        detection_a = False
        detection_b = False
        
        max_conf_a = -1
        max_conf_category_a = ''
        
        max_conf_b = -1
        max_conf_category_b = ''
        
        # det = im_a['detections'][0]
        for det in im_a['detections']:
            category_id = det['category']
            conf = det['conf']
            if conf < max_conf_a:
                continue
            else:
                max_conf_a = conf
                max_conf_category_a = category_id
            if conf >= options.detection_thresholds_a[detection_categories[category_id]]:
                detection_a = True
                break
            
        for det in im_b['detections']:
            category_id = det['category']
            conf = det['conf']
            if conf < max_conf_b:
                continue
            else:
                max_conf_b = conf
                max_conf_category_b = category_id
            if conf >= options.detection_thresholds_b[detection_categories[category_id]]:
                detection_b = True
                break
    
        im_pair = (im_a,im_b)
        
        if detection_a and detection_b:
            if max_conf_category_a == max_conf_category_b:
                common_detections[fn] = im_pair
            else:
                class_transitions[fn] = im_pair
        elif (not detection_a) and (not detection_b):
            common_non_detections[fn] = im_pair
        elif detection_a and (not detection_b):
            detections_a_only[fn] = im_pair
        else:
            assert detection_b and (not detection_a)
            detections_b_only[fn] = im_pair
            
    # ...for each filename
    
    print('Of {} files:\n{} common detections\n{} common non-detections\n{} A only\n{} B only\n{} class transitions'.format(
        len(filenames_a),len(common_detections),
        len(common_non_detections),len(detections_a_only),
        len(detections_b_only),len(class_transitions)))
        
    
    ##%% Sample and plot differences
    
    from multiprocessing.pool import ThreadPool
    
    if options.n_rendering_threads > 1:
       print('Rendering images with {} workers'.format(options.n_rendering_threads))
       pool = ThreadPool(options.n_rendering_threads)    
    
    
    categories_to_image_pairs = {
        'common_detections':common_detections,
        'common_non_detections':common_non_detections,
        'detections_a_only':detections_a_only,
        'detections_b_only':detections_b_only,
        'class_transitions':class_transitions
    }
    
    def render_detection_comparisons(category,image_pairs,image_filnames):
        
        print('Rendering detections for category {}'.format(category))
        
        category_folder = os.path.join(options.output_folder,category)
        os.makedirs(category_folder,exist_ok=True)
        
        def render_image_pair(fn):
            
            input_image_path = os.path.join(options.image_folder,fn)
            assert os.path.isfile(input_image_path), 'Image {} does not exist'.format(input_image_path)
            
            im = visualization_utils.open_image(input_image_path)
            image_pair = image_pairs[fn]
            detections_a = image_pair[0]['detections']
            detections_b = image_pair[1]['detections']
            
            """
            def render_detection_bounding_boxes(detections, image,
                                                label_map={},
                                                classification_label_map={},
                                                confidence_threshold=0.8, thickness=4, expansion=0,
                                                classification_confidence_threshold=0.3,
                                                max_classifications=3,
                                                colormap=COLORS):
            """
            if options.target_width is not None:
                im = visualization_utils.resize_image(im, options.target_width)
                
            visualization_utils.render_detection_bounding_boxes(detections_a,im,
                                                                confidence_threshold=options.rendering_confidence_threshold_a,
                                                                thickness=4,expansion=0,
                                                                colormap=options.colormap_a,
                                                                textalign=visualization_utils.TEXTALIGN_LEFT)
            visualization_utils.render_detection_bounding_boxes(detections_b,im,
                                                                confidence_threshold=options.rendering_confidence_threshold_b,
                                                                thickness=2,expansion=0,
                                                                colormap=options.colormap_b,
                                                                textalign=visualization_utils.TEXTALIGN_RIGHT)
        
            output_image_fn = path_utils.flatten_path(fn)
            output_image_path = os.path.join(category_folder,output_image_fn)
            im.save(output_image_path)           
            return output_image_path
        
        # ...def render_image_pair()
        
        # fn = image_filenames[0]
        if options.n_rendering_threads <= 1:
            output_image_paths = []
            for fn in tqdm(image_filenames):        
                output_image_paths.append(render_image_pair(fn))
        else:
            output_image_paths = list(tqdm(pool.imap(render_image_pair, image_filenames), total=len(image_filenames)))
        
        return output_image_paths
    
    # ...def render_detection_comparisons()
    
    # category = 'common_detections'
    for category in categories_to_image_pairs.keys():
        
        # Choose detection pairs we're going to render for this category
        image_pairs = categories_to_image_pairs[category]
        image_filenames = list(image_pairs.keys())
        if len(image_filenames) > options.max_images_per_category:
            print('Sampling {} of {} image pairs for category {}'.format(
                options.max_images_per_category,
                len(image_filenames),
                category))
            image_filenames = random.sample(image_filenames,
                                            options.max_images_per_category)
        assert len(image_filenames) <= options.max_images_per_category
        category_image_paths = render_detection_comparisons(category,
                                                            image_pairs,image_filenames)
        
        category_html_filename = os.path.join(options.output_folder,
                                              category + '.html')
        category_image_paths_relative = [os.path.relpath(s,
                                                         options.output_folder) for s in category_image_paths]
        
        image_info = []
        for fn in category_image_paths_relative:
            info = {
                'filename': fn,
                'title': fn,
                'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
            }
            image_info.append(info)
    
        write_html_image_list(
            category_html_filename,
            images=image_info,
            options={
                'headerHtml': '<h1>{}</h1>'.format(category)
            })
        
    # ...for each category
    
    
    ##%% Write the top-level HTML file
    
    html_output_file = os.path.join(options.output_folder,'index.html')
    
    style_header = """<head>
        <style type="text/css">
        a { text-decoration: none; }
        body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
        div.contentdiv { margin-left: 20px; }
        </style>
        </head>"""
    
    index_page = ''
    index_page += '<html>\n{}\n<body>\n'.format(style_header)
    index_page += '<h2>Comparison of results for {}</h2>\n'.format(
        options.job_name)
    index_page += '<p>Comparing <b>{}</b> (A, red) to <b>{}</b> (B, blue)</p>'.format(
        options.results_description_a,options.results_description_b)
    index_page += '<div class="contentdiv">\n'
    index_page += 'Detection thresholds for {}:\n{}<br/>'.format(
        options.results_description_a,str(options.detection_thresholds_a))
    index_page += 'Detection thresholds for {}:\n{}<br/>'.format(
        options.results_description_b,str(options.detection_thresholds_b))
    index_page += 'Rendering threshold for {}:\n{}<br/>'.format(
        options.results_description_a,str(options.rendering_confidence_threshold_a))
    index_page += 'Rendering threshold for {}:\n{}<br/>'.format(
        options.results_description_b,str(options.rendering_confidence_threshold_b))
    
    index_page += '<br/>'
    
    index_page += 'Rendering a maximum of {} images per category<br/>'.format(options.max_images_per_category)
    
    index_page += '<br/>'
    
    index_page += ('Of {} total files:<br/><br/><div style="margin-left:15px;">{} common detections<br/>{} common non-detections<br/>{} A only<br/>{} B only<br/>{} class transitions</div><br/>'.format(
        len(filenames_a),len(common_detections),
        len(common_non_detections),len(detections_a_only),
        len(detections_b_only),len(class_transitions)))
    
    index_page += 'Comparison pages:<br/><br/>\n'
    index_page += '<div style="margin-left:15px;">\n'
    
    for category in categories_to_image_pairs.keys():
        category_html_filename = category + '.html'
        index_page += '<a href="{}">{}</a><br/>\n'.format(
            category_html_filename,category)
    
    index_page += '</div>'    
    index_page += '</div></body></html>\n'
    
    with open(html_output_file,'w') as f:
        f.write(index_page) 

    results = BatchComparisonResults()
    results.html_output_file = html_output_file
    return results
    
# ...def compare_batch_results()


#%% Interactive driver

if False:
    
    #%%
    
    options = BatchComparisonOptions()
    options.output_folder = os.path.expanduser('~/tmp/kru-comparison')
    
    options.image_folder = os.path.expanduser('~/data/KRU')
    
    options.results_filename_a = os.path.expanduser('~/postprocessing/snapshot-safari/snapshot-safari-2022-04-07/combined_api_outputs/snapshot-safari-2022-04-07_detections.kru-only.json')
    # options.results_filename_b = os.path.expanduser('~/postprocessing/snapshot-safari/snapshot-safari-mdv5_camonly-5a-2022-04-12/combined_api_outputs/snapshot-safari-mdv5_camonly-5a-2022-04-12_detections.json')
    options.results_filename_b = os.path.expanduser('~/postprocessing/snapshot-safari/snapshot-safari-mdv5_torchscript_camonly-2022-04-14/combined_api_outputs/snapshot-safari-mdv5_torchscript_camonly-2022-04-14_detections.json')
    
    options.job_name = 'KRU'
    options.results_description_a = 'MDv4'
    options.results_description_b = 'MDv5-camonly-28-torchscript'
    
    options.detection_thresholds_a = {'animal':0.7,'person':0.7,'vehicle':0.7}
    options.detection_thresholds_b = {'animal':0.4,'person':0.4,'vehicle':0.4}
    
    results = compare_batch_results(options)
    
    open_file(results.html_output_file)
    
    #%% 
    
    options = BatchComparisonOptions()
    options.output_folder = os.path.expanduser('~/tmp/ffi-comparison')
    
    options.image_folder = os.path.expanduser('~/data/ffi/deployment')
    
    options.results_filename_a = os.path.expanduser('~/postprocessing/ffi/ffi-2022-02-09/combined_api_outputs/ffi-2022-02-09_detections.json')
    options.results_filename_b = os.path.expanduser('~/postprocessing/ffi/ffi-camonly-torschscript-28-2022-04-13/combined_api_outputs/ffi-camonly-torschscript-28-2022-04-13_detections.json')
    
    options.job_name = 'FFI'
    options.results_description_a = 'MDv4'
    options.results_description_b = 'MDv5-camonly-28-torchscript'
    
    options.detection_thresholds_a = {'animal':0.7,'person':0.7,'vehicle':0.7}
    options.detection_thresholds_b = {'animal':0.4,'person':0.4,'vehicle':0.4}
    
    results = compare_batch_results(options)
    
    open_file(results.html_output_file)
    
    
    