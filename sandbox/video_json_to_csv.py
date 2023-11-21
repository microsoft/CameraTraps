#
# Takes a .json file with MD results for an individual video, and converts to a .csv that includes 
# frame times.  This is very bespoke to animal detection and does not include other classes.
#

#%% Imports and constants

import json
from detection import video_utils

# Only necessary if you want to extract the sample rate from the video
video_file = r"G:\temp\wpz\1_fps_20211216_101100.mp4"

results_file = r"G:\temp\wpz\1_fps_20211216_101100.mp4.json"
output_file = results_file.replace('.json','.animals.csv')
Fs = None


#%% Extract the sample rate if necessary

if Fs is None:
    
    Fs = video_utils.get_video_fs(video_file)
    
    
#%% Load results

with open(results_file,'r') as f:
    results = json.load(f)
    
print('Loaded results for {} frames'.format(len(results['images'])))


#%% Convert to .csv

detection_categories = results['detection_categories']
animal_category = [d for d in detection_categories.keys() if detection_categories[d] == 'animal'][0]
assert animal_category == '1'
n_detections_to_include = 4

with open(output_file,'w') as f:
    
    f.write('time (seconds),')
    for i_detection in range(0,n_detections_to_include):
        f.write('detection_{}_x,detection_{}_x,detection_{}_height,detection_{}_width,detection_{}_confidence'.format(
            i_detection,i_detection,i_detection,i_detection,i_detection))
        if i_detection != n_detections_to_include:
            f.write(',')
    f.write('\n')
    
    # i_image = 0; im = results['images'][i_image]
    for i_image,im in enumerate(results['images']):
        detections = im['detections']
        animal_detections = [d for d in detections if d['category'] == animal_category]
        sorted_detections = sorted(animal_detections, key = lambda d: d['conf'], reverse=True)
        
        ts = i_image * Fs
        f.write(str(ts) + ',')
        for i_detection in range(0,n_detections_to_include):
            if i_detection >= len(sorted_detections):
                x = ''
                y = ''
                conf = ''
            else:
                bbox = sorted_detections[i_detection]['bbox']
                xf = bbox[0] + bbox[2]/2.0; x = str(xf)
                yf = bbox[1] + bbox[3]/2.0; y = str(yf)
                wf = bbox[2]; w = str(wf)
                hf = bbox[3]; h = str(hf)
                conf = str(sorted_detections[i_detection]['conf'])
            f.write('{},{},{},{},{}'.format(x,y,w,h,conf))
            if i_detection != n_detections_to_include - 1:
                f.write(',')
        f.write('\n')
                
                
                