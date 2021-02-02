######
#
# video_utils.py
#
# Utilities for splitting, rendering, and assembling videos.
#
######

#%% Constants, imports, environment

import os
import cv2
import glob

# from ai4eutils
import path_utils
    
from tqdm import tqdm
from typing import Container,Iterable,List

from visualization import visualization_utils as vis_utils


#%% Path utilities

# TODO: migrate into ai4eutils/path_utils if this is sticky

VIDEO_EXTENSIONS = ('.mp4','.avi','.mpeg','.mpg')

def is_video_file(s: str, video_extensions: Container[str] = VIDEO_EXTENSIONS
                  ) -> bool:
    """
    Checks a file's extension against a hard-coded set of video file
    extensions.
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in video_extensions


def find_video_strings(strings: Iterable[str]) -> List[str]:
    """
    Given a list of strings that are potentially video file names, looks for
    strings that actually look like video file names (based on extension).
    """
    return [s for s in strings if is_video_file(s)]


def find_videos(dirname: str, recursive: bool = False) -> List[str]:
    """
    Finds all files in a directory that look like video file names. Returns
    absolute paths.
    """
    if recursive:
        strings = glob.glob(os.path.join(dirname, '**', '*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirname, '*.*'))
    return find_video_strings(strings)


#%% Function for rendering frames to video and vice-versa

# http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html

def frames_to_video(images, Fs, output_file_name, codec_spec='h264'):
    """
    Given a list of image files and a sample rate, concatenate those images into
    a video and write to [output_file_name].
    """
    
    if len(images) == 0:
        return

    # Determine the width and height from the first image
    frame = cv2.imread(images[0])
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, Fs, (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def get_video_fs(input_video_file):
    assert os.path.isfile(input_video_file), 'File {} not found'.format(input_video_file)    
    vidcap = cv2.VideoCapture(input_video_file)
    Fs = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return Fs

# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
def video_to_frames(input_video_file, output_folder):
    """
    Render every frame of [input_video_file] to a .jpg in [output_folder]
    """
    
    assert os.path.isfile(input_video_file), 'File {} not found'.format(input_video_file)
    
    vidcap = cv2.VideoCapture(input_video_file)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    Fs = vidcap.get(cv2.CAP_PROP_FPS)
    print('Reading {} frames at {} Hz from {}'.format(n_frames,Fs,input_video_file))

    frame_filenames = []

    for frame_number in tqdm(range(0,n_frames)):

        success,image = vidcap.read()
        if not success:
            print('Read terminating at frame {} of {}'.format(frame_number,n_frames))
            break

        frame_filename = 'frame{:05d}.jpg'.format(frame_number)
        frame_filename = os.path.join(output_folder,frame_filename)
        frame_filenames.append(frame_filename)
        try:
            cv2.imwrite(frame_filename,image)
            assert os.path.isfile(frame_filename)
        except KeyboardInterrupt:
            vidcap.release()
            raise
        except Exception as e:
            print('Error on frame {} of {}: {}'.format(frame_number,n_frames,str(e)))

    print('\nExtracted {} of {} frames'.format(len(frame_filenames),n_frames))

    vidcap.release()    
    return frame_filenames,Fs


def video_folder_to_frames(input_folder:str, output_folder_base:str, recursive:bool=True):
    """
    For every video file in input_folder, create a folder within output_folder_base, and 
    render every frame of the video to .jpg in that folder.
    """
    
    # Recursively enumerate video files
    input_files_full_paths = find_videos(input_folder,recursive=True)
    input_files_relative_paths = [os.path.relpath(s,input_folder) for s in input_files_full_paths]
    input_files_relative_paths = [s.replace('\\','/') for s in input_files_relative_paths]
    
    os.makedirs(output_folder_base,exist_ok=True)    
    
    frame_filenames_by_video = []
    fs_by_video = []
    
    # For each video
    # input_fn_relative = input_files_relative_paths[0]
    for i_video,input_fn_relative in enumerate(input_files_relative_paths):
        
        print('Processing video {} of {}'.format(i_video,len(input_files_relative_paths)))
        
        input_fn_absolute = os.path.join(input_folder,input_fn_relative)
        assert os.path.isfile(input_fn_absolute)
    
        # Create the target output folder
        output_folder_video = os.path.join(output_folder_base,input_fn_relative)
        os.makedirs(output_folder_video,exist_ok=True)
    
        # Render frames
        frame_filenames,fs = video_to_frames(input_fn_absolute,output_folder_video)
        frame_filenames_by_video.append(frame_filenames)
        fs_by_video.append(fs)
    
    return frame_filenames_by_video,fs_by_video
    
    
#%% Test driver

if False:

    #%% Constants
    
    Fs = 30.01
    confidence_threshold = 0.75
    input_folder = 'z:\\'
    frame_folder_base = r'e:\video_test\frames'
    detected_frame_folder_base = r'e:\video_test\detected_frames'
    rendered_videos_folder_base = r'e:\video_test\rendered_videos'
    
    results_file = r'results.json'
    os.makedirs(detected_frame_folder_base,exist_ok=True)
    os.makedirs(rendered_videos_folder_base,exist_ok=True)
    
    
    #%% Split videos into frames
        
    frame_filenames_by_video,fs_by_video = video_folder_to_frames(input_folder,frame_folder_base,recursive=True)
    
    
    #%% List image files, break into folders
    
    frame_files = path_utils.find_images(frame_folder_base,True)
    frame_files = [s.replace('\\','/') for s in frame_files]
    print('Enumerated {} total frames'.format(len(frame_files)))
    
    Fs = 30.01
    # Find unique folders
    folders = set()
    # fn = frame_files[0]
    for fn in frame_files:
        folders.add(os.path.dirname(fn))
    folders = [s.replace('\\','/') for s in folders]
    print('Found {} folders for {} files'.format(len(folders),len(frame_files)))
    
    
    
    #%% Load detector output
    
    import json
    
    with open(results_file,'r') as f:
        detection_results = json.load(f)
    detections = detection_results['images']
    detector_label_map = detection_results['detection_categories']
    for d in detections:
        d['file'] = d['file'].replace('\\','/').replace('video_frames/','')


    #%% Render detector frames
    
    # folder = list(folders)[0]
    for folder in folders:
        
        frame_files_this_folder = [fn for fn in frame_files if folder in fn]
        folder_relative = folder.replace((frame_folder_base + '/').replace('\\','/'),'')
        detection_results_this_folder = [d for d in detections if folder_relative in d['file']]
        print('Found {} detections in folder {}'.format(len(detection_results_this_folder),folder))
        assert len(frame_files_this_folder) == len(detection_results_this_folder)
        
        rendered_frame_output_folder = os.path.join(detected_frame_folder_base,folder_relative)
        os.makedirs(rendered_frame_output_folder,exist_ok=True)
        
        # d = detection_results_this_folder[0]
        for d in tqdm(detection_results_this_folder):
            
            input_file = os.path.join(frame_folder_base,d['file'])
            output_file = os.path.join(detected_frame_folder_base,d['file'])
            os.makedirs(os.path.dirname(output_file),exist_ok=True)
            vis_utils.draw_bounding_boxes_on_file(input_file,output_file,d['detections'],
                                                  confidence_threshold)
        
        # ...for each file in this folder
            
    # ...for each folder


    #%% Render output videos
            
    # folder = list(folders)[0]
    for folder in tqdm(folders):
        
        folder_relative = folder.replace((frame_folder_base + '/').replace('\\','/'),'')
        rendered_detector_output_folder = os.path.join(detected_frame_folder_base,folder_relative)
        assert os.path.isdir(rendered_detector_output_folder)
        
        frame_files_relative = os.listdir(rendered_detector_output_folder)
        frame_files_absolute = [os.path.join(rendered_detector_output_folder,s) for s in frame_files_relative]
        
        output_video_filename = os.path.join(rendered_videos_folder_base,folder_relative)
        os.makedirs(os.path.dirname(output_video_filename),exist_ok=True)
        
        original_video_filename = output_video_filename.replace(rendered_videos_folder_base,input_folder)
        assert os.path.isfile(original_video_filename)
        Fs = get_video_fs(original_video_filename)
                
        frames_to_video(frame_files_absolute, Fs, output_video_filename)