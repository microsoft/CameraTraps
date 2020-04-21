######
#
# process_video.py
#
# Split a video into frames, run the frames through run_tf_detector_batch.py, and
# optionally stitch together results into a new video with detection boxes.
#
######

#%% Constants, imports, environment

import os
import tempfile
import shutil
import argparse

from tqdm import tqdm

import cv2

from detection import run_tf_detector_batch 
from visualization import visualize_detector_output
from ct_utils import args_to_object


#%% Function for rendering frames to video and vice-versa

# http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html    

def frames_to_video(images,Fs,output_file_name):
    
    if len(images) == 0:
        return
    
    # Determine the width and height from the first image    
    frame = cv2.imread(images[0])
    cv2.imshow('video',frame)
    height, width, channels = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_file_name, fourcc, Fs, (width, height))
    
    for iImage,image in enumerate(images):
    
        frame = cv2.imread(images[iImage])    
        out.write(frame)
        
    out.release()
    cv2.destroyAllWindows()

    
# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames    
    
def video_to_frames(input_video_file,output_folder):
    
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
            raise
        except Exception as e:
            print('Error on frame {} of {}: {}'.format(frame_number,n_frames,str(e)))
        
    print('\nExtracted {} of {} frames'.format(len(frame_filenames),n_frames))
    
    return frame_filenames,Fs
    

#%% Main function
    
class ProcessVideoOptions:
    
    model_file = ''    
    input_video_file = ''
    
    output_json_file = None
    output_video_file = None
    
    render_output_video = False
    delete_output_frames = True
    
    confidence_threshold = 0.8
    n_cores = 1
    
    debug_max_frames = -1
        
    
def process_video(options):
    
    if options.output_json_file is None:
        options.output_json_file = options.input_video_file + '.json'
        
    if options.render_output_video and (options.output_video_file is None):
        options.output_video_file = options.input_video_file + '.detections.mp4'
    
    temp_folder = os.path.join(tempfile.gettempdir(),'process_camera_trap_video')
    frame_output_folder = os.path.join(temp_folder,os.path.basename(options.input_video_file) + '_frames')
    os.makedirs(frame_output_folder,exist_ok=True)
        
    frame_filenames,Fs = video_to_frames(options.input_video_file,frame_output_folder)
    
    image_file_names = frame_filenames
    if options.debug_max_frames > 0:
        image_file_names = image_file_names[0:options.debug_max_frames]
    
    results = run_tf_detector_batch.load_and_run_detector_batch(options.model_file,image_file_names,
                                          confidence_threshold=options.confidence_threshold, 
                                          n_cores=options.n_cores) 
    
    run_tf_detector_batch.write_results_to_file(results,options.output_json_file,
                                                relative_path_base=frame_output_folder)
        
    
    if options.render_output_video:
        
        # Render detections to images
        viz_options = visualize_detector_output.DetectorVizOptions()
        viz_options.detector_output_path = options.output_json_file
        viz_options.images_dir = frame_output_folder    
        rendering_output_dir = os.path.join(temp_folder,os.path.basename(options.input_video_file) + '_detections')
        viz_options.out_dir = rendering_output_dir    
        detected_frame_files = visualize_detector_output.visualize_detector_output(viz_options)        
        
        # Combine into a video
        frames_to_video(detected_frame_files,Fs,options.output_video_file)
    
    if options.delete_output_frames:
        shutil.rmtree(temp_folder)

    return results


#%% Interactive driver
   
if False:
    
    #%% Load video and split into frames
    
    model_file = r'c:\temp\models\md_v4.0.0.pb'
    input_video_file = r'C:\temp\LIFT0003.MP4'
    
    options = ProcessVideoOptions()
    options.model_file = model_file
    options.input_video_file = input_video_file
    options.debug_max_frames = 10
    
    process_video(options)
    
    # python process_video.py "c:\temp\models\md_v4.0.0.pb" "c:\temp\LIFT0003.MP4" --debug_max_frames=10 --render_output_video=True
    

#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description=('Run the MegaDetector each frame in a video, optionally producing a new video with detections annotated'))
    
    parser.add_argument('model_file', type=str,
                        help='MegaDetector model file')
    
    parser.add_argument('input_video_file', type=str,
                        help='video file to process')
    
    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json')
    
    parser.add_argument('--output_video_file', type=str,
                        default=None, help='video output file, defaults to [video file].mp4')
    
    parser.add_argument('--render_output_video', type=bool,
                        default=False, help='enable/disable video output rendering (default False)')
    
    parser.add_argument('--delete_output_frames', type=bool,
                        default=False, help='enable/disable temporary file detection (default True)')
    
    parser.add_argument('--confidence_threshold', type=float,
                        default=0.8, help="dont render boxes with confidence below this threshold")
    
    parser.add_argument('--n_cores', type=int,
                        default=1, help='number of cores to use for detection (CPU only)')
    
    parser.add_argument('--debug_max_frames', type=int,
                        default=-1, help='trim to N frames for debugging')
        
    args = parser.parse_args()
    options = ProcessVideoOptions()
    args_to_object(args,options)
    
    process_video(options)

if __name__ == '__main__':
    main()
