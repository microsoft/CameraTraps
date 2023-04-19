#
# aggregate_video.py
#
# Aggregate results and render output video for a video that's already been run through MD
#

#%% Constants

import os
import json
from detection.process_video import ProcessVideoOptions
from detection.video_utils import frames_to_video, video_to_frames
from visualization import visualize_detector_output

video_file = os.path.expanduser('~/1_fps_20211216_101100.mp4')
results_file_raw = os.path.expanduser('~/1_fps_20211216_101100.mp4.json')
allowed_categories = ['1']
max_box_size = 0.15
min_box_size = 0.03

results_file = results_file_raw.replace('.json','_animals_only.json')


#%% Processing        

with open(results_file_raw,'r') as f:

    n_detections = 0
    n_valid_detections = 0
    min_valid_confidence = 1.0
    
    d = json.load(f)
    # im = d['images'][0]
    
    for im in d['images']:
        
        valid_detections = []
        max_detection_conf = 0
        
        for det in im['detections']:
            n_detections += 1
            det_size = None
            if 'bbox' in det:
                bbox = det['bbox']
                det_size = bbox[2] * bbox[3]
            if det['category'] in allowed_categories and \
                ((det_size is not None) and (det_size < max_box_size) and \
                 (det_size > min_box_size)):
                if det['conf'] > max_detection_conf:
                    max_detection_conf = det['conf']
                if det['conf'] < min_valid_confidence:
                    min_valid_confidence = det['conf']
                valid_detections.append(det)
                n_valid_detections += 1
                
        # ...for each detection
        
        im['detections'] = valid_detections
        
        # This is no longer included in output files by default
        if 'max_detection_conf' in im:
            im['max_detection_conf'] = max_detection_conf
        
    print('Kept {} of {} detections (min conf {})'.format(
        n_valid_detections,n_detections,min_valid_confidence))
    
    with open(results_file,'w') as f:
        json.dump(d,f,indent=2)
        
base_dir = os.path.expanduser('~/frame_processing')
min_confidence = 0.001

assert os.path.isfile(video_file) and os.path.isfile(results_file)
os.makedirs(base_dir,exist_ok=True)


## Split into frames

options = ProcessVideoOptions()
options.input_video_file = video_file
options.output_json_file = results_file
options.output_video_file = options.input_video_file + '.detections.mp4'

frame_output_folder = os.path.join(
    base_dir, os.path.basename(options.input_video_file) + '_frames')
if os.path.isdir(frame_output_folder):
    print('Frame output folder exists, skipping frame extraction')
else:
    os.makedirs(frame_output_folder, exist_ok=True)

    frame_filenames, Fs = video_to_frames(
        options.input_video_file, frame_output_folder)
        
## Render output video

### Render detections to images

rendering_output_dir = os.path.join(
    base_dir, os.path.basename(options.input_video_file) + '_detections')
os.makedirs(rendering_output_dir,exist_ok=True)
detected_frame_files = visualize_detector_output.visualize_detector_output(
    detector_output_path=options.output_json_file,
    out_dir=rendering_output_dir,
    images_dir=frame_output_folder,
    confidence=min_confidence)

### Combine into a video

Fs = 20
frames_to_video(detected_frame_files, Fs, options.output_video_file)

