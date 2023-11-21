######
#
# process_video.py
#
# Split a video (or folder of videos) into frames, run the frames through run_detector_batch.py,
# and optionally stitch together results into a new video with detection boxes.
#
# TODO: allow video rendering when processing a whole folder
# TODO: allow video rendering from existing results
#
######

#%% Constants, imports, environment

import os
import tempfile
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
    
    # Only relevant if render_output_video is True
    output_video_file = None
    
    # Folder to use for extracted frames
    frame_folder = None
    
    # Folder to use for rendered frames (if rendering output video)
    frame_rendering_folder = None
    
    # Should we render a video with detection boxes?
    #
    # Only supported when processing a single video, not a folder.
    render_output_video = False
    
    # If we are rendering boxes to a new video, should we keep the temporary
    # rendered frames?
    keep_rendered_frames = False
    
    # Should we keep the extracted frames?
    keep_extracted_frames = False
    
    reuse_results_if_available = False
    recursive = False 
    verbose = False

    rendering_confidence_threshold = 0.15
    json_confidence_threshold = 0.005
    frame_sample = None
    
    n_cores = 1

    debug_max_frames = -1


#%% Main functions

def process_video(options):
    """
    Process a single video
    """

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
        
    # TODO: keep track of whether we created this folder, delete if we're deleting the extracted
    # frames and we created the folder, and the output files aren't in the same folder.  For now,
    # we're just deleting the extracted frames and leaving the empty folder around in this case.
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
            n_cores=options.n_cores,
            quiet=(not options.verbose))
    
        run_detector_batch.write_results_to_file(
            results, options.output_json_file,
            relative_path_base=frame_output_folder,
            detector_file=options.model_file)

    if options.render_output_video:
        
        # Render detections to images
        if options.frame_rendering_folder is not None:
            rendering_output_dir = options.frame_rendering_folder
        else:
            rendering_output_dir = os.path.join(
                tempdir, os.path.basename(options.input_video_file) + '_detections')
            
        # TODO: keep track of whether we created this folder, delete if we're deleting the rendered
        # frames and we created the folder, and the output files aren't in the same folder.  For now,
        # we're just deleting the rendered frames and leaving the empty folder around in this case.
        os.makedirs(rendering_output_dir,exist_ok=True)
        
        detected_frame_files = visualize_detector_output.visualize_detector_output(
            detector_output_path=options.output_json_file,
            out_dir=rendering_output_dir,
            images_dir=frame_output_folder,
            confidence_threshold=options.rendering_confidence_threshold)

        # Combine into a video
        print('Rendering video at {} fps'.format(Fs))
        frames_to_video(detected_frame_files, Fs, options.output_video_file)
        
        # Delete the temporary directory we used for detection images
        if not options.keep_rendered_frames:
            try:
                # shutil.rmtree(rendering_output_dir)
                for rendered_frame_fn in detected_frame_files:
                    os.remove(rendered_frame_fn)
            except Exception as e:
                print('Warning: error deleting rendered frames from folder {}:\n{}'.format(
                    rendering_output_dir,str(e)))
                pass
        
    # (Optionally) delete the frames on which we ran MegaDetector
    if not options.keep_extracted_frames:
        try:
            # shutil.rmtree(frame_output_folder)
            for extracted_frame_fn in frame_filenames:
                os.remove(extracted_frame_fn)
        except Exception as e:
            print('Warning: error extracted frames from folder {}:\n{}'.format(
                frame_output_folder,str(e)))
            pass
        
    return results

# ...process_video()


def process_video_folder(options):
    """
    Process a folder of videos    
    """
    
    ## Validate options

    assert not options.render_output_video, \
        'Video rendering is not supported when rendering a folder'
       
    assert os.path.isdir(options.input_video_file), \
        '{} is not a folder'.format(options.input_video_file)
           
    assert options.output_json_file is not None, \
        'When processing a folder, you must specify an output .json file'
                         
    assert options.output_json_file.endswith('.json')
    video_json = options.output_json_file
    frames_json = options.output_json_file.replace('.json','.frames.json')
    os.makedirs(os.path.dirname(video_json),exist_ok=True)
    
    
    ## Split every video into frames
    
    if options.frame_folder is not None:
        frame_output_folder = options.frame_folder
    else:
        tempdir = os.path.join(tempfile.gettempdir(), 'process_camera_trap_video')
        frame_output_folder = os.path.join(
            tempdir, os.path.basename(options.input_video_file) + '_frames_' + str(uuid1()))
    os.makedirs(frame_output_folder, exist_ok=True)

    print('Extracting frames')
    frame_filenames, Fs, video_filenames = \
        video_folder_to_frames(input_folder=options.input_video_file,
                               output_folder_base=frame_output_folder, 
                               recursive=options.recursive, overwrite=True,
                               n_threads=options.n_cores,every_n_frames=options.frame_sample,
                               verbose=options.verbose)
    
    image_file_names = list(itertools.chain.from_iterable(frame_filenames))
    
    if len(image_file_names) == 0:
        if len(video_filenames) == 0:
            print('No videos found in folder {}'.format(options.input_video_file))
        else:
            print('No frames extracted from folder {}, this may be due to an '\
                  'unsupported video codec'.format(options.input_video_file))
        return

    if options.debug_max_frames is not None and options.debug_max_frames > 0:
        image_file_names = image_file_names[0:options.debug_max_frames]
    
    
    ## Run MegaDetector on the extracted frames
    
    if options.reuse_results_if_available and \
        os.path.isfile(frames_json):
            print('Loading results from {}'.format(frames_json))
            results = None
    else:
        print('Running MegaDetector')
        results = run_detector_batch.load_and_run_detector_batch(
            options.model_file, image_file_names,
            confidence_threshold=options.json_confidence_threshold,
            n_cores=options.n_cores,
            quiet=(not options.verbose))
    
        run_detector_batch.write_results_to_file(
            results, frames_json,
            relative_path_base=frame_output_folder,
            detector_file=options.model_file)
    
    
    ## (Optionally) delete the frames on which we ran MegaDetector
    
    if not options.keep_extracted_frames:
        try:
            print('Deleting frame cache')
            # shutil.rmtree(frame_output_folder)
            for frame_fn in image_file_names:
                os.remove(frame_fn)
        except Exception as e:
            print('Warning: error deleting frames from folder {}:\n{}'.format(
                frame_output_folder,str(e)))
            pass
    

    ## Convert frame-level results to video-level results

    print('Converting frame-level results to video-level results')
    frame_results_to_video_results(frames_json,video_json)

# ...process_video_folder()


def options_to_command(options):
    
    cmd = 'python process_video.py'
    cmd += ' "' + options.model_file + '"'
    cmd += ' "' + options.input_video_file + '"'
    
    if options.recursive:
        cmd += ' --recursive'
    if options.frame_folder is not None:
        cmd += ' --frame_folder' + ' "' + options.frame_folder + '"'
    if options.frame_rendering_folder is not None:
        cmd += ' --frame_rendering_folder' + ' "' + options.frame_rendering_folder + '"'    
    if options.output_json_file is not None:
        cmd += ' --output_json_file' + ' "' + options.output_json_file + '"'
    if options.output_video_file is not None:
        cmd += ' --output_video_file' + ' "' + options.output_video_file + '"'
    if options.keep_extracted_frames:
        cmd += ' --keep_extracted_frames'
    if options.reuse_results_if_available:
        cmd += ' --reuse_results_if_available'    
    if options.render_output_video:
        cmd += ' --render_output_video'
    if options.keep_rendered_frames:
        cmd += ' --keep_rendered_frames'    
    if options.rendering_confidence_threshold is not None:
        cmd += ' --rendering_confidence_threshold ' + str(options.rendering_confidence_threshold)
    if options.json_confidence_threshold is not None:
        cmd += ' --json_confidence_threshold ' + str(options.json_confidence_threshold)
    if options.n_cores is not None:
        cmd += ' --n_cores ' + str(options.n_cores)
    if options.frame_sample is not None:
        cmd += ' --frame_sample ' + str(options.frame_sample)
    if options.debug_max_frames is not None:
        cmd += ' --debug_max_frames ' + str(options.debug_max_frames)

    return cmd

    
#%% Interactive driver


if False:    
    
    #%% Process a folder of videos
    
    model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    input_dir = r'g:\temp\test\input-video'
    frame_folder = r'g:\temp\test\extracted-frames'
    rendering_folder = r'g:\temp\test\rendered-frames'
    output_json_file = r'g:\temp\test\output.json'
    
    print('Processing folder {}'.format(input_dir))
    
    options = ProcessVideoOptions()    
    options.model_file = model_file
    options.input_video_file = input_dir
    options.frame_folder = frame_folder
    options.output_json_file = output_json_file
    options.frame_rendering_folder = rendering_folder
    
    cmd = options_to_command(options)
    print(cmd)
    # import clipboard; clipboard.copy(cmd)
    
    if False:
        process_video_folder(options)
        
    
    #%% Process a single video

    model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')    
    input_video_file = r"G:\temp\test\input-video\test.AVI"
    frame_folder = r'g:\temp\test\extracted-frames'
    rendering_folder = r'g:\temp\test\rendered-frames'
    
    options = ProcessVideoOptions()
    options.model_file = model_file
    options.input_video_file = input_video_file
    options.frame_folder = frame_folder
    options.frame_rendering_folder = rendering_folder
    options.render_output_video = True
    
    cmd = options_to_command(options)
    print(cmd)
    # import clipboard; clipboard.copy(cmd)
    
    if False:        
        process_video(options)
    
    
    #%% Render a folder of videos, one file at a time

    model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    input_dir = r'g:\temp\test\input-video'
    frame_folder = r'g:\temp\test\extracted-frames'
    rendering_folder = r'g:\temp\test\rendered-frames'
    
    print('Processing folder {}'.format(input_dir))
    
    options = ProcessVideoOptions()    
    options.model_file = model_file
    options.frame_folder = frame_folder
    options.frame_rendering_folder = rendering_folder
    options.render_output_video = True
    options.rendering_confidence_threshold = 0.05
    
    from detection import video_utils
    video_files = video_utils.find_videos(input_dir)
    video_files = [fn for fn in video_files if 'detections' not in fn]

    commands = []    
    for video_fn in video_files:
        options.input_video_file = video_fn    
        cmd = options_to_command(options)
        commands.append(cmd)
        
    s = '\n\n'.join(commands)
    print(s)
    # import clipboard; clipboard.copy(s)
    
            
    
#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description=(
        'Run MegaDetector on each frame in a video (or every Nth frame), optionally '\
        'producing a new video with detections annotated'))

    parser.add_argument('model_file', type=str,
                        help='MegaDetector model file')

    parser.add_argument('input_video_file', type=str,
                        help='video file (or folder) to process')

    parser.add_argument('--recursive', action='store_true',
                        help='recurse into [input_video_file]; only meaningful if a folder '\
                         'is specified as input')
    
    parser.add_argument('--frame_folder', type=str, default=None,
                        help='folder to use for intermediate frame storage, defaults to a folder '\
                        'in the system temporary folder')
        
    parser.add_argument('--frame_rendering_folder', type=str, default=None,
                        help='folder to use for rendered frame storage, defaults to a folder in '\
                        'the system temporary folder')
    
    parser.add_argument('--output_json_file', type=str,
                        default=None, help='.json output file, defaults to [video file].json')

    parser.add_argument('--output_video_file', type=str,
                        default=None, help='video output file (or folder), defaults to '\
                            '[video file].mp4 for files, or [video file]_annotated] for folders')

    parser.add_argument('--keep_extracted_frames',
                       action='store_true', help='Disable the deletion of extracted frames')
    
    parser.add_argument('--reuse_results_if_available',
                       action='store_true', help='If the output .json files exists, and this flag is set,'\
                           'we\'ll skip running MegaDetector')
    
    parser.add_argument('--render_output_video', action='store_true',
                        help='enable video output rendering (not rendered by default)')

    parser.add_argument('--keep_rendered_frames',
                       action='store_true', help='Disable the deletion of rendered (w/boxes) frames')

    parser.add_argument('--rendering_confidence_threshold', type=float,
                        default=0.8, help="don't render boxes with confidence below this threshold")

    parser.add_argument('--json_confidence_threshold', type=float,
                        default=0.0, help="don't include boxes in the .json file with confidence '\
                            'below this threshold")

    parser.add_argument('--n_cores', type=int,
                        default=1, help='number of cores to use for frame separation and detection. '\
                            'If using a GPU, this option will be respected for frame separation but '\
                            'ignored for detection.  Only relevant to frame separation when processing '\
                            'a folder.')

    parser.add_argument('--frame_sample', type=int,
                        default=None, help='procss every Nth frame (defaults to every frame)')

    parser.add_argument('--debug_max_frames', type=int,
                        default=-1, help='trim to N frames for debugging (impacts model execution, '\
                            'not frame rendering)')

    args = parser.parse_args()
    options = ProcessVideoOptions()
    args_to_object(args,options)

    if os.path.isdir(options.input_video_file):
        process_video_folder(options)
    else:
        process_video(options)

if __name__ == '__main__':
    main()
