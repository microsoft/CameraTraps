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

from detection import run_tf_detector_batch
from visualization import visualize_detector_output
from ct_utils import args_to_object
from detection.video_utils import video_to_frames
from detection.video_utils import frames_to_video
from detection.video_utils import find_videos


#%% Main function

class ProcessVideoOptions:

    model_file = ''
    input_video_file = ''

    output_json_file = None
    output_video_file = None

    render_output_video = False
    delete_output_frames = True
    reuse_results_if_available = False

    confidence_threshold = 0.8
    n_cores = 1

    debug_max_frames = -1


def process_video(options):

    if options.output_json_file is None:
        options.output_json_file = options.input_video_file + '.json'

    if options.render_output_video and (options.output_video_file is None):
        options.output_video_file = options.input_video_file + '.detections.mp4'

    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    frame_output_folder = os.path.join(
        tempdir, os.path.basename(options.input_video_file) + '_frames')
    os.makedirs(frame_output_folder, exist_ok=True)

    frame_filenames, Fs = video_to_frames(
        options.input_video_file, frame_output_folder)

    image_file_names = frame_filenames
    if options.debug_max_frames > 0:
        image_file_names = image_file_names[0:options.debug_max_frames]

    if options.reuse_results_if_available and \
        os.path.isfile(options.output_json_file):
            print('Loading results from {}'.format(options.output_json_file))
            results = None
    else:
        results = run_tf_detector_batch.load_and_run_detector_batch(
            options.model_file, image_file_names,
            confidence_threshold=options.confidence_threshold,
            n_cores=options.n_cores)
    
        run_tf_detector_batch.write_results_to_file(
            results, options.output_json_file,
            relative_path_base=frame_output_folder)


    if options.render_output_video:
        # Render detections to images
        rendering_output_dir = os.path.join(
            tempdir, os.path.basename(options.input_video_file) + '_detections')
        detected_frame_files = visualize_detector_output.visualize_detector_output(
            detector_output_path=options.output_json_file,
            out_dir=rendering_output_dir,
            images_dir=frame_output_folder)

        # Combine into a video
        frames_to_video(detected_frame_files, Fs, options.output_video_file)

    if options.delete_output_frames:
        shutil.rmtree(tempdir)

    return results


#%% Interactive driver


if False:

    #%% Process a folder of videos
    
    import re

    model_file = r'c:\temp\models\md_v4.1.0.pb'
    input_dir = r'C:\temp\bellevue_test\videos'
    
    video_files = find_videos(input_dir)
    video_files = [s for s in video_files if 'detections' not in s]
    
    # input_video_file = video_files[1]
    for input_video_file in video_files:
        
        print('Processing file {}'.format(input_video_file))
        
        options = ProcessVideoOptions()
        options.reuse_results_if_available = True
        options.model_file = model_file
        options.input_video_file = input_video_file
        options.output_video_file = re.sub('.avi','.detections.mp4',input_video_file,flags=re.I)
        options.output_json_file = re.sub('.avi','.detections.json',input_video_file,flags=re.I)
        options.render_output_video = True
        options.n_cores = 1
        options.confidence_threshold = 0.55
        options.delete_output_frames = True
        process_video(options)
        
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
