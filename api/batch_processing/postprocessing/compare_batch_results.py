"""
compare_batch_results.py

Compare two sets of batch results; typically used to compare MegaDetector versions.

Currently supports only detection results (not classification results).
"""

#%% Imports

import json
import os
import random
import copy

from tqdm import tqdm
from visualization import visualization_utils

# Assumes ai4eutils is on the python path (https://github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list
import path_utils


#%% Constants and support classes

# We will confirm that this matches what we load from each file
detection_categories = {'1': 'animal', '2': 'person', '3': 'vehicle'}
    
class PairwiseBatchComparisonOptions:
    
    results_filename_a = None
    results_filename_b = None
    
    results_description_a = None
    results_description_b = None
    
    detection_thresholds_a = {'animal':0.7,'person':0.7,'vehicle':0.7}
    detection_thresholds_b = {'animal':0.7,'person':0.7,'vehicle':0.7}

    rendering_confidence_threshold_a = 0.1
    rendering_confidence_threshold_b = 0.1

class BatchComparisonOptions:
    
    output_folder = None
    image_folder = None    
    
    max_images_per_category = 1000
    colormap_a = ['Red']
    colormap_b = ['RoyalBlue']

    target_width = 800    
    n_rendering_threads = 50        
    random_seed = 0    
    
    pairwise_options = PairwiseBatchComparisonOptions()
    
    
class BatchComparisonResults:
    
    html_output_file = None
    

main_page_style_header = """<head>
    <style type="text/css">
    a { text-decoration: none; }
    body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
    div.contentdiv { margin-left: 20px; }
    </style>
    </head>"""

main_page_header = '<html>\n{}\n<body>\n'.format(main_page_style_header)
main_page_footer = '<br/><br/><br/></body></html>\n'


#%% Main function

def _compare_batch_results(options,output_index,pairwise_options):
        
    assert options.pairwise_options is None
    
    random.seed(options.random_seed)

    ##%% Validate inputs
    
    assert os.path.isfile(pairwise_options.results_filename_a)
    assert os.path.isfile(pairwise_options.results_filename_b)
    assert os.path.isdir(options.image_folder)
    os.makedirs(options.output_folder,exist_ok=True)
    
    
    ##%% Load both result sets
    
    with open(pairwise_options.results_filename_a,'r') as f:
        results_a = json.load(f)
    
    with open(pairwise_options.results_filename_b,'r') as f:
        results_b = json.load(f)
        
    assert results_a['detection_categories'] == detection_categories
    assert results_b['detection_categories'] == detection_categories
    
    if pairwise_options.results_description_a is None:
        if 'detector' not in results_a['info']:
            print('No model metadata supplied for results-A, assuming MDv4')
            pairwise_options.results_description_a = 'MDv4 (assumed)'
        else:            
            pairwise_options.results_description_a = results_a['info']['detector']
    
    if pairwise_options.results_description_b is None:
        if 'detector' not in results_b['info']:
            print('No model metadata supplied for results-B, assuming MDv4')
            pairwise_options.results_description_b = 'MDv4 (assumed)'
        else:            
            pairwise_options.results_description_b = results_b['info']['detector']
    
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
        
        categories_above_threshold_a = set()

        if not 'detections' in im_a:
            assert 'failure' in im_a and im_a['failure'] is not None
            continue
        
        if not 'detections' in im_b:
            assert 'failure' in im_b and im_b['failure'] is not None
            continue
        
        invalid_category_error = False
        
        # det = im_a['detections'][0]
        for det in im_a['detections']:
            
            category_id = det['category']
            
            if category_id not in detection_categories:
                print('Warning: unexpected category {} for model A on file {}'.format(category_id,fn))
                invalid_category_error = True
                break
                
            conf = det['conf']
            
            if conf >= pairwise_options.detection_thresholds_a[detection_categories[category_id]]:
                categories_above_threshold_a.add(category_id)
                            
        if invalid_category_error:
            continue
        
        categories_above_threshold_b = set()
        
        for det in im_b['detections']:
            
            if category_id not in detection_categories:
                print('Warning: unexpected category {} for model B on file {}'.format(category_id,fn))
                invalid_category_error = True
                break
            
            category_id = det['category']
            conf = det['conf']
            
            if conf >= pairwise_options.detection_thresholds_b[detection_categories[category_id]]:
                categories_above_threshold_b.add(category_id)
                            
        if invalid_category_error:
            continue
        
        im_pair = (im_a,im_b)
        
        detection_a = (len(categories_above_threshold_a) > 0)
        detection_b = (len(categories_above_threshold_b) > 0)
                
        if detection_a and detection_b:            
            if categories_above_threshold_a == categories_above_threshold_b:
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
    
    categories_to_page_titles = {
        'common_detections':'Detections common to both models',
        'common_non_detections':'Non-detections common to both models',
        'detections_a_only':'Detections reported by model A only',
        'detections_b_only':'Detections reported by model B only',
        'class_transitions':'Detections reported as different classes by models A and B'
    }

    local_output_folder = os.path.join(options.output_folder,'cmp_' + \
                                       str(output_index).zfill(3))

    def render_detection_comparisons(category,image_pairs,image_filenames):
        
        print('Rendering detections for category {}'.format(category))
                
        category_folder = os.path.join(local_output_folder,category)
        os.makedirs(category_folder,exist_ok=True)
        
        # Render two sets of results (i.e., a comparison) for a single
        # image.
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
                                                                confidence_threshold=pairwise_options.rendering_confidence_threshold_a,
                                                                thickness=4,expansion=0,
                                                                colormap=options.colormap_a,
                                                                textalign=visualization_utils.TEXTALIGN_LEFT)
            visualization_utils.render_detection_bounding_boxes(detections_b,im,
                                                                confidence_threshold=pairwise_options.rendering_confidence_threshold_b,
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
    
    # For each category, generate comparison images and the 
    # comparison HTML page.
    #
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

        input_image_absolute_paths = [os.path.join(options.image_folder,fn) for fn in image_filenames]
        
        category_image_output_paths = render_detection_comparisons(category,
                                                            image_pairs,image_filenames)
        
        category_html_filename = os.path.join(local_output_folder,
                                              category + '.html')
        category_image_output_paths_relative = [os.path.relpath(s,local_output_folder) \
                                         for s in category_image_output_paths]
        
        image_info = []
        
        assert len(category_image_output_paths_relative) == len(input_image_absolute_paths)
        
        import urllib
        for i_fn,fn in enumerate(category_image_output_paths_relative): 
            info = {
                'filename': fn,
                'title': fn,
                'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5',
                'linkTarget': urllib.parse.quote(input_image_absolute_paths[i_fn])
            }
            image_info.append(info)
    
        category_page_header_string = '<h1>{}</h1>'.format(categories_to_page_titles[category])
        category_page_header_string += '<p style="font-weight:bold;">\n'
        category_page_header_string += 'Model A: {}<br/>\n'.format(pairwise_options.results_description_a)
        category_page_header_string += 'Model B: {}'.format(pairwise_options.results_description_b)
        category_page_header_string += '</p>\n'
        
        category_page_header_string += '<p>\n'
        category_page_header_string += 'Detection thresholds for A ({}):\n{}<br/>'.format(
            pairwise_options.results_description_a,str(pairwise_options.detection_thresholds_a))
        category_page_header_string += 'Detection thresholds for B ({}):\n{}<br/>'.format(
            pairwise_options.results_description_b,str(pairwise_options.detection_thresholds_b))
        category_page_header_string += 'Rendering threshold for A ({}):\n{}<br/>'.format(
            pairwise_options.results_description_a,str(pairwise_options.rendering_confidence_threshold_a))
        category_page_header_string += 'Rendering threshold for B ({}):\n{}<br/>'.format(
            pairwise_options.results_description_b,str(pairwise_options.rendering_confidence_threshold_b))
        category_page_header_string += '</p>\n'        
        
        write_html_image_list(
            category_html_filename,
            images=image_info,
            options={
                'headerHtml': category_page_header_string
            })
        
    # ...for each category
    
    
    ##%% Write the top-level HTML file content

    html_output_string  = ''
    
    html_output_string += '<p>Comparing <b>{}</b> (A, red) to <b>{}</b> (B, blue)</p>'.format(
        pairwise_options.results_description_a,pairwise_options.results_description_b)
    html_output_string += '<div class="contentdiv">\n'
    html_output_string += 'Detection thresholds for {}:\n{}<br/>'.format(
        pairwise_options.results_description_a,str(pairwise_options.detection_thresholds_a))
    html_output_string += 'Detection thresholds for {}:\n{}<br/>'.format(
        pairwise_options.results_description_b,str(pairwise_options.detection_thresholds_b))
    html_output_string += 'Rendering threshold for {}:\n{}<br/>'.format(
        pairwise_options.results_description_a,str(pairwise_options.rendering_confidence_threshold_a))
    html_output_string += 'Rendering threshold for {}:\n{}<br/>'.format(
        pairwise_options.results_description_b,str(pairwise_options.rendering_confidence_threshold_b))
    
    html_output_string += '<br/>'
    
    html_output_string += 'Rendering a maximum of {} images per category<br/>'.format(options.max_images_per_category)
    
    html_output_string += '<br/>'
    
    html_output_string += ('Of {} total files:<br/><br/><div style="margin-left:15px;">{} common detections<br/>{} common non-detections<br/>{} A only<br/>{} B only<br/>{} class transitions</div><br/>'.format(
        len(filenames_a),len(common_detections),
        len(common_non_detections),len(detections_a_only),
        len(detections_b_only),len(class_transitions)))
    
    html_output_string += 'Comparison pages:<br/><br/>\n'
    html_output_string += '<div style="margin-left:15px;">\n'
        
    comparison_path_relative = os.path.relpath(local_output_folder,options.output_folder)    
    for category in categories_to_image_pairs.keys():
        category_html_filename = os.path.join(comparison_path_relative,category + '.html')
        html_output_string += '<a href="{}">{}</a><br/>\n'.format(
            category_html_filename,category)
    
    html_output_string += '</div>\n'
    html_output_string += '</div>\n'
    
    return html_output_string
        
# ...def compare_batch_results()

def compare_batch_results(options):
    
    assert options.pairwise_options is not None    
    options = copy.deepcopy(options)
 
    if not isinstance(options.pairwise_options,list):
        options.pairwise_options = [options.pairwise_options]
    
    pairwise_options_list = options.pairwise_options
    n_comparisons = len(pairwise_options_list)
    
    options.pairwise_options = None
    
    html_content = ''
        
    for i_comparison,pairwise_options in enumerate(pairwise_options_list):
        print('Running comparison {} of {}'.format(i_comparison,n_comparisons))
        html_content += _compare_batch_results(options,i_comparison,pairwise_options)
            
    html_output_string = main_page_header
    html_output_string += '<h2>Comparison of results for {}</h2>\n'.format(
        options.job_name)
    html_output_string += html_content
    html_output_string += main_page_footer
    
    html_output_file = os.path.join(options.output_folder,'index.html')    
    with open(html_output_file,'w') as f:
        f.write(html_output_string) 
    
    results = BatchComparisonResults()
    results.html_output_file = html_output_file
    return results


#%% Interactive driver

if False:
    
    #%% KRU
    
    options = BatchComparisonOptions()
    options.output_folder = os.path.expanduser('~/tmp/kru-comparison')
    options.job_name = 'KRU'    
    options.image_folder = os.path.expanduser('~/data/KRU')
    
    pairwise_options = PairwiseBatchComparisonOptions()
    
    pairwise_options.results_filename_a = os.path.expanduser('~/postprocessing/snapshot-safari/snapshot-safari-2022-04-07/combined_api_outputs/snapshot-safari-2022-04-07_detections.kru-only.json')
    pairwise_options.results_filename_b = os.path.expanduser('~/postprocessing/snapshot-safari/snapshot-safari-mdv5_torchscript_camonly-2022-04-14/combined_api_outputs/snapshot-safari-mdv5_torchscript_camonly-2022-04-14_detections.json')
    
    pairwise_options.detection_thresholds_a = {'animal':0.7,'person':0.7,'vehicle':0.7}
    pairwise_options.detection_thresholds_b = {'animal':0.4,'person':0.4,'vehicle':0.4}
    
    options.pairwise_options = pairwise_options
    
    results = compare_batch_results(options)
    
    path_utils.open_file(results.html_output_file)
    

#%% Command-line driver

## TODO
    