######
#
# process_video.py
#
# Split a video into frames, run the frames through run_detector_batch.py, and
# optionally stitch together results into a new video with detection boxes.
#
######

#%% Constants, imports, environment

import os
import tempfile
import shutil
import argparse
import itertools

from detection import run_detector_batch
from visualization import visualize_detector_output
from ct_utils import args_to_object
from detection.video_utils import video_to_frames
from detection.video_utils import frames_to_video
from detection.video_utils import frame_results_to_video_results
from detection.video_utils import video_folder_to_frames
from uuid import uuid1

class ProcessVideoOptions:

    model_file = ''
    input_video_file = ''

    output_json_file = None
    output_video_file = None
    
    frame_folder = None
    rendering_folder = None
    
    render_output_video = False
    keep_output_frames = False
    reuse_results_if_available = False
    recursive = False 

    rendering_confidence_threshold = 0.8
    json_confidence_threshold = 0.0
    frame_sample = None
    
    n_cores = 1

    debug_max_frames = -1


#%% Main function

def process_video(options):

    if options.output_json_file is None:
        options.output_json_file = options.input_video_file + '.json'

    if options.render_output_video and (options.output_video_file is None):
        options.output_video_file = options.input_video_file + '.detections.mp4'

    tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
    
    if options.frame_folder is not None:
        frame_output_folder = options.frame_folder
    else:
        frame_output_folder = os.path.join(
            tempdir, os.path.basename(options.input_video_file) + '_frames_' + str(uuid1()))
    os.makedirs(frame_output_folder, exist_ok=True)

    frame_filenames, Fs = video_to_frames(
        options.input_video_file, frame_output_folder, every_n_frames=options.frame_sample)

    image_file_names = frame_filenames
    if options.debug_max_frames > 0:
        image_file_names = image_file_names[0:options.debug_max_frames]

    if options.reuse_results_if_available and \
        os.path.isfile(options.output_json_file):
            print('Loading results from {}'.format(options.output_json_file))
            results = None
    else:
        results = run_detector_batch.load_and_run_detector_batch(
            options.model_file, image_file_names,
            confidence_threshold=options.json_confidence_threshold,
            n_cores=options.n_cores)
    
        run_detector_batch.write_results_to_file(
            results, options.output_json_file,
            relative_path_base=frame_output_folder)

    if options.render_output_video:
        
        # Render detections to images
        if options.rendering_folder is not None:
            rendering_output_dir = options.rendering_folder
        else:
            rendering_output_dir = os.path.join(
                tempdir, os.path.basename(options.input_video_file) + '_detections')
        detected_frame_files = visualize_detector_output.visualize_detector_output(
            detector_output_path=options.output_json_file,
            out_dir=rendering_output_dir,
            images_dir=frame_output_folder,
            confidence=options.rendering_confidence_threshold)

        # Combine into a video
        print('Rendering video at {} fps'.format(Fs))
        frames_to_video(detected_frame_files, Fs, options.output_video_file)
        
        # Delete the temporary directory we used for detection images
        if not options.keep_output_frames:
            try:
                shutil.rmtree(rendering_output_dir)
            except Exception:
                pass
        
    # (Optionally) delete the frames on which we ran MegaDetector
    if not options.keep_output_frames:
        try:
            shutil.rmtree(frame_output_folder)
        except Exception:
            pass
        
    return results


def process_video_folder(options):
    
    ## Validate options

    assert not options.render_output_video, 'Video rendering is not supported when rendering a folder'

    
    ## Split every video into frames
    
    assert os.path.isdir(options.input_video_file),'{} is not a folder'.format(options.input_video_file)
                
    if options.frame_folder is not None:
        frame_output_folder = options.frame_folder
    else:
        tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
        frame_output_folder = os.path.join(
            tempdir, os.path.basename(options.input_video_file) + '_frames_' + str(uuid1()))
    os.makedirs(frame_output_folder, exist_ok=True)

    frame_filenames, Fs = video_folder_to_frames(input_folder=options.input_video_file, output_folder_base=frame_output_folder, 
                               recursive=options.recursive, overwrite=True,
                               n_threads=options.n_cores,every_n_frames=options.frame_sample)
    
    image_file_names = list(itertools.chain.from_iterable(frame_filenames))

    if options.debug_max_frames is not None and options.debug_max_frames > 0:
        image_file_names = image_file_names[0:options.debug_max_frames]
    
    
    ## Run MegaDetector
    
    if options.output_json_file is None:
        frames_json = options.input_video_file + '.frames.json'
        video_json = options.input_video_file + '.json'
    else:
        if '.json' in options.output_json_file:
            frames_json = options.output_json_file.replace('.json','.frames.json')
            video_json = options.output_json_file
        else:
            video_json = options.output_json_file
            frames_json = video_json + '_frames'
            
    if options.reuse_results_if_available and \
        os.path.isfile(frames_json):
            print('Loading results from {}'.format(frames_json))
            results = None
    else:
        results = run_detector_batch.load_and_run_detector_batch(
            options.model_file, image_file_names,
            confidence_threshold=options.json_confidence_threshold,
            n_cores=options.n_cores)
    
        run_detector_batch.write_results_to_file(
            results, frames_json,
            relative_path_base=frame_output_folder)
    
    
    ## Convert frame-level results to video-level results

    frame_results_to_video_results(frames_json,video_json)
         
        
#%% Interactive driver


if False:

    #%% Process a folder of videos
    
    model_file = r'g:\temp\models\md_v4.1.0.pb'
    input_dir = r'g:\temp\100MEDIA_mp4'
    frame_folder = r'g:\temp\frames'
    
    print('Processing folder {}'.format(input_dir))
    
    options = ProcessVideoOptions()
    options.reuse_results_if_available = True
    options.model_file = model_file
    options.input_video_file = input_dir
    options.output_video_file = None
    options.output_json_file = None
    options.frame_folder = frame_folder
    options.render_output_video = False
    options.n_cores = 5
    options.json_confidence_threshold = 0.55
    options.delete_output_frames = False
    options.recursive = True
    options.debug_max_frames = None
    options.frame_sample = 10
    
    # process_video_folder(options)
    
    cmd = 'python process_video.py'
    cmd += ' "' + model_file + '"'
    cmd += ' "' + input_dir + '"'
    if options.recursive:
        cmd += ' --recursive'
    if options.frame_folder is not None:
        cmd += ' --frame_folder' + ' "' + options.frame_folder + '"'
    if options.render_output_video:
        cmd += ' --render_output_video'
    if options.delete_output_frames:
        cmd += ' --delete_output_frames'
    cmd += ' --rendering_confidence_threshold ' + str(options.rendering_confidence_threshold)
    cmd += ' --json_confidence_threshold ' + str(options.json_confidence_threshold)
    cmd += ' --n_cores ' + str(options.n_cores)
    if options.frame_sample is not None:
        cmd += ' --frame_sample ' + str(options.frame_sample)
    if options.debug_max_frames is not None:
        cmd += ' --debug_max_frames ' + str(options.debug_max_frames)

    # import clipboard; clipboard.copy(cmd)

    
    #%% Process a single video

    model_file = r'c:\temp\models\md_v4.0.0.pb'
    input_video_file = r'C:\temp\LIFT0003.MP4'

    options = ProcessVideoOptions()
    options.model_file = model_file
    options.input_video_file = input_video_file
    options.debug_max_frames = 10

    process_video(options)

    
    # python process_video.py "c:\temp\models\md_v4.0.0.pb" "c:\temp\LIFT0003.MP4" --debug_max_frames=10 --render_output_video=True


    #%% For a video that's already been run through MD
    
    import json
    video_file = os.path.expanduser('~/1_fps_20211216_101100.mp4')
    results_file_raw = os.path.expanduser('~/1_fps_20211216_101100.mp4.json')
    allowed_categories = ['1']
    max_box_size = 0.15
    min_box_size = 0.03
    
    results_file = results_file_raw.replace('.json','_animals_only.json')
            
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
                if det['category'] in allowed_categories and ((det_size is not None) and (det_size < max_box_size) and (det_size > min_box_size)):
                    if det['conf'] > max_detection_conf:
                        max_detection_conf = det['conf']
                    if det['conf'] < min_valid_confidence:
                        min_valid_confidence = det['conf']
                    valid_detections.append(det)
                    n_valid_detections += 1
                    
            # ...for each detection
            
            im['detections'] = valid_detections
            im['max_detection_conf'] = max_detection_conf
            
        print('Kept {} of {} detections (min conf {})'.format(n_valid_detections,n_detections,min_valid_confidence))
        
        with open(results_file,'w') as f:
            json.dump(d,f,indent=2)
            
    base_dir = os.path.expanduser('~/frame_processing')
    min_confidence = 0.001
    
    assert os.path.isfile(video_file) and os.path.isfile(results_file)
    os.makedirs(base_dir,exist_ok=True)
    
    
    # Split into frames
    
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
            
    # Render output video
    
    ## Render detections to images
    
    rendering_output_dir = os.path.join(
        base_dir, os.path.basename(options.input_video_file) + '_detections')
    os.makedirs(rendering_output_dir,exist_ok=True)
    detected_frame_files = visualize_detector_output.visualize_detector_output(
        detector_output_path=options.output_json_file,
        out_dir=rendering_output_dir,
        images_dir=frame_output_folder,
        confidence=min_confidence)

    ## Combine into a video
    
    Fs = 20
    frames_to_video(detected_frame_files, Fs, options.output_video_file)


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description=('Run MegaDetector on each frame in a video (or every Nth frame), optionally producing a new video with detections annotated'))

    parser.add_argument('model_file', type=str,
                        help='MegaDetector model file')

    parser.add_argument('input_video_file', type=str,
                        help='video file (or folder) to process')

    parser.add_argument('--recursive', action='store_true',
                        help='recurse into [input_video_file]; only meaningful if a folder is specified as input')
    
    parser.add_argument('--frame_folder', type=str, default=None,
                        help='folder to use for intermediate frame storage, defaults to a folder in the system temporary folder')
                        
    parser.add_argument('--rendering_folder', type=str, default=None,
                        help='folder to use for renderred frame storage, defaults to a folder in the system temporary folder')
    
    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json')

    parser.add_argument('--output_video_file', type=str,
                        default=None, help='video output file (or folder), defaults to [video file].mp4 for files, or [video file]_annotated] for folders')

    parser.add_argument('--render_output_video', action='store_true',
                        help='enable video output rendering (not rendered by default)')

    parser.add_argument('--keep_output_frames',
                       action='store_true', help='Disable the deletion of intermediate images (pre- and post-detection rendered frames)')

    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold")

    parser.add_argument('--json_confidence_threshold', type=float,
                        default=0.0, help="don't include boxes in the .json file with confidence below this threshold")

    parser.add_argument('--n_cores', type=int,
                        default=1, help='number of cores to use for detection (CPU only)')

    parser.add_argument('--frame_sample', type=int,
                        default=None, help='procss every Nth frame (defaults to every frame)')

    parser.add_argument('--debug_max_frames', type=int,
                        default=-1, help='trim to N frames for debugging (impacts model execution, not frame rendering)')

    args = parser.parse_args()
    options = ProcessVideoOptions()
    args_to_object(args,options)

    if os.path.isdir(options.input_video_file):
        process_video_folder(options)
    else:
        process_video(options)

if __name__ == '__main__':
    main()
