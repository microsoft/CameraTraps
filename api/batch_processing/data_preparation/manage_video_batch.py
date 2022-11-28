#
# manage_video_batch.py
#
# Notebook-esque script to manage the process of running a local batch of videos
# through MD.  Defers most of the heavy lifting to manage_local_batch.py .
#

#%% Imports and constants

import path_utils
import os
from detection import video_utils

input_folder = '/datadrive/data'
output_folder_base = '/datadrive/frames'

assert os.path.isdir(input_folder)
os.makedirs(output_folder_base,exist_ok=True)


#%% Split videos into frames

assert os.path.isdir(input_folder)
os.makedirs(output_folder_base,exist_ok=True)

recursive = True
overwrite = True
n_threads = 5
every_n_frames = 10

frame_filenames_by_video,fs_by_video = video_utils.video_folder_to_frames(input_folder=input_folder,
                                                              output_folder_base=output_folder_base,
                                                              recursive=recursive,
                                                              overwrite=overwrite,
                                                              n_threads=n_threads,
                                                              every_n_frames=every_n_frames)


#%% List image files, break into folders

from collections import defaultdict

frame_files = path_utils.find_images(output_folder_base,True)
frame_files = [s.replace('\\','/') for s in frame_files]
print('Enumerated {} total frames'.format(len(frame_files)))

# Find unique (relative) folders
folder_to_frame_files = defaultdict(list)

# fn = frame_files[0]
for fn in frame_files:
    folder_name = os.path.dirname(fn)
    folder_name = os.path.relpath(folder_name,output_folder_base)
    folder_to_frame_files[folder_name].append(fn)

print('Found {} folders for {} files'.format(len(folder_to_frame_files),len(frame_files)))


#%% List videos

video_filenames = video_utils.find_videos(input_folder,recursive=True)
video_filenames = [os.path.relpath(fn,input_folder) for fn in video_filenames]
print('Input folder contains {} videos'.format(len(video_filenames)))


#%% Check for videos that are missing entirely

# list(folder_to_frame_files.keys())[0]
# video_filenames[0]

missing_videos = []

# fn = video_filenames[0]
for relative_fn in video_filenames:
    if relative_fn not in folder_to_frame_files:
        missing_videos.append(relative_fn)
        
print('{} of {} folders are missing frames entirely'.format(len(missing_videos),
                                                            len(video_filenames)))


#%% Check for videos with very few frames

min_frames_for_valid_video = 10

low_frame_videos = []

for folder_name in folder_to_frame_files.keys():
    frame_files = folder_to_frame_files[folder_name]
    if len(frame_files) < min_frames_for_valid_video:
        low_frame_videos.append(folder_name)

print('{} of {} folders have fewer than {} frames'.format(
    len(low_frame_videos),len(video_filenames),min_frames_for_valid_video))


#%% Print the list of videos that are problematic

print('Videos that could not be decoded:\n')

for fn in missing_videos:
    print(fn)
    
print('\nVideos with fewer than {} decoded frames:\n'.format(min_frames_for_valid_video))

for fn in low_frame_videos:
    print(fn)
    
    
#%% Process images like we would for any other camera trap job

# ...typically using manage_local_batch.py, but do this however you like, as long
# as you get a results file at the end.
#
# If you do RDE, remember to use the second folder from the bottom, rather than the
# bottom-most folder.


#%% Convert frame results to video results
    
from detection.video_utils import frame_results_to_video_results

filtered_output_filename = '/results/organization/stuff.json'
video_output_filename = filtered_output_filename.replace('.json','_aggregated.json')
frame_results_to_video_results(filtered_output_filename,video_output_filename)
    

#%% Confirm that the videos in the .json file are what we expect them to be

import json

with open(video_output_filename,'r') as f:
    video_results = json.load(f)

video_filenames_set = set(video_filenames)

filenames_in_video_results_set = set([im['file'] for im in video_results['images']])

for fn in filenames_in_video_results_set:
    assert fn in video_filenames_set
    

#%% Scrap

if False:
    
    pass

    #%% Test a possibly-broken video
    
    fn = '/datadrive/tmp/video.AVI'
    
    
    fs = video_utils.get_video_fs(fn)
    print(fs)
    
    tmpfolder = '/home/user/tmp/frametmp'
    os.makedirs(tmpfolder,exist_ok=True)
    
    video_utils.video_to_frames(fn, tmpfolder, verbose=True, every_n_frames=10)
    
    
    #%% List videos in a folder
    
    input_folder = '/datadrive/tmp/organization/data'
    video_filenames = video_utils.find_videos(input_folder,recursive=True)
