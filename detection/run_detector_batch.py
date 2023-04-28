r"""
Module to run an animal detection model on lots of images, writing the results
to a file in the same format produced by our batch API:

https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing

This enables the results to be used in our post-processing pipeline; see
api/batch_processing/postprocessing/postprocess_batch_results.py .

This script can save results to checkpoints intermittently, in case disaster
strikes. To enable this, set --checkpoint_frequency to n > 0, and results 
will be saved as a checkpoint every n images. Checkpoints will be written 
to a file in the same directory as the output_file, and after all images
are processed and final results file written to output_file, the temporary
checkpoint file will be deleted. If you want to resume from a checkpoint, set
the checkpoint file's path using --resume_from_checkpoint.

The `threshold` you can provide as an argument is the confidence threshold above
which detections will be included in the output file.

Has preliminary multiprocessing support for CPUs only; if a GPU is available, it will
use the GPU instead of CPUs, and the --ncores option will be ignored.  Checkpointing
is not supported when using multiprocessing.

Does not have a command-line option to bind the process to a particular GPU, but you can 
prepend with "CUDA_VISIBLE_DEVICES=0 ", for example, to bind to GPU 0, e.g.:

CUDA_VISIBLE_DEVICES=0 python detection/run_detector_batch.py ~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb ~/data/test-small ~/tmp/mdv4test.json --output_relative_filenames --recursive

"""

#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import shutil
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly
from tqdm import tqdm

# from multiprocessing.pool import ThreadPool as workerpool
import multiprocessing
from threading import Thread
from multiprocessing import Process
from multiprocessing.pool import Pool as workerpool

# Number of images to pre-fetch
max_queue_size = 10
use_threads_for_queue = False
verbose = False

# Useful hack to force CPU inference.
#
# Need to do this before any PT/TF imports, which happen when we import
# from run_detector.
force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import detection.run_detector as run_detector
from detection.run_detector import ImagePathUtils,\
    is_gpu_available,\
    load_detector,\
    get_detector_version_from_filename,\
    get_detector_metadata_from_version_string

import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)


#%% Support functions for multiprocessing

def producer_func(q,image_files):
    """ 
    Producer function; only used when using the (optional) image queue.
    
    Reads up to N images from disk and puts them on the blocking queue for processing.
    """
    
    if verbose:
        print('Producer starting'); sys.stdout.flush()
        
    for im_file in image_files:
    
        try:
            if verbose:
                print('Loading image {}'.format(im_file)); sys.stdout.flush()
            image = viz_utils.load_image(im_file)
        except Exception as e:
            print('Producer process: image {} cannot be loaded. Exception: {}'.format(im_file, e))
            raise
        
        if verbose:
            print('Queueing image {}'.format(im_file)); sys.stdout.flush()
        q.put([im_file,image])                    
    
    q.put(None)
        
    print('Finished image loading'); sys.stdout.flush()
    
    
def consumer_func(q,return_queue,model_file,confidence_threshold,image_size=None):
    """ 
    Consumer function; only used when using the (optional) image queue.
    
    Pulls images from a blocking queue and processes them.
    """
    
    if verbose:
        print('Consumer starting'); sys.stdout.flush()

    start_time = time.time()
    detector = load_detector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model (before queueing) in {}'.format(humanfriendly.format_timespan(elapsed)))
    sys.stdout.flush()
        
    results = []
    
    n_images_processed = 0
    
    while True:
        r = q.get()
        if r is None:
            q.task_done()
            return_queue.put(results)
            return
        n_images_processed += 1
        im_file = r[0]
        image = r[1]
        if verbose or ((n_images_processed % 10) == 0):
            elapsed = time.time() - start_time
            images_per_second = n_images_processed / elapsed
            print('De-queued image {} ({:.2f}/s) ({})'.format(n_images_processed,
                                                          images_per_second,
                                                          im_file));
            sys.stdout.flush()
        results.append(process_image(im_file=im_file,detector=detector,
                                     confidence_threshold=confidence_threshold,
                                     image=image,quiet=True,image_size=image_size))
        if verbose:
            print('Processed image {}'.format(im_file)); sys.stdout.flush()
        q.task_done()
            

def run_detector_with_image_queue(image_files,model_file,confidence_threshold,
                                  quiet=False,image_size=None):
    """
    Driver function for the (optional) multiprocessing-based image queue; only used 
    when --use_image_queue is specified.  Starts a reader process to read images from disk, but 
    processes images in the  process from which this function is called (i.e., does not currently
    spawn a separate consumer process).
    """
    
    q = multiprocessing.JoinableQueue(max_queue_size)
    return_queue = multiprocessing.Queue(1)
    
    if use_threads_for_queue:
        producer = Thread(target=producer_func,args=(q,image_files,))
    else:
        producer = Process(target=producer_func,args=(q,image_files,))
    producer.daemon = False
    producer.start()
 
    # TODO
    #
    # The queue system is a little more elegant if we start one thread for reading and one
    # for processing, and this works fine on Windows, but because we import TF at module load,
    # CUDA will only work in the main process, so currently the consumer function runs here.
    #
    # To enable proper multi-GPU support, we may need to move the TF import to a separate module
    # that isn't loaded until very close to where inference actually happens.
    run_separate_consumer_process = False

    if run_separate_consumer_process:
        if use_threads_for_queue:
            consumer = Thread(target=consumer_func,args=(q,return_queue,model_file,
                                                         confidence_threshold,image_size,))
        else:
            consumer = Process(target=consumer_func,args=(q,return_queue,model_file,
                                                          confidence_threshold,image_size,))
        consumer.daemon = True
        consumer.start()
    else:
        consumer_func(q,return_queue,model_file,confidence_threshold,image_size)

    producer.join()
    print('Producer finished')
   
    if run_separate_consumer_process:
        consumer.join()
        print('Consumer finished')
    
    q.join()
    print('Queue joined')

    results = return_queue.get()
    
    return results


#%% Other support funtions

def chunks_by_number_of_chunks(ls, n):
    """
    Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]


#%% Image processing functions

def process_images(im_files, detector, confidence_threshold, use_image_queue=False, 
                   quiet=False, image_size=None):
    """
    Runs MegaDetector over a list of image files.

    Args
    - im_files: list of str, paths to image files
    - detector: loaded model or str (path to .pb/.pt model file)
    - confidence_threshold: float, only detections above this threshold are returned

    Returns
    - results: list of dict, each dict represents detections on one image
        see the 'images' key in https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    
    if isinstance(detector, str):
        start_time = time.time()
        detector = load_detector(detector)
        elapsed = time.time() - start_time
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    if use_image_queue:
        run_detector_with_image_queue(im_files, detector, confidence_threshold, 
                                      quiet=quiet, image_size=image_size)
    else:
        results = []
        for im_file in im_files:
            results.append(process_image(im_file, detector, confidence_threshold,
                                         quiet=quiet, image_size=image_size))
        return results
    

def process_image(im_file, detector, confidence_threshold, image=None, 
                  quiet=False, image_size=None):
    """
    Runs MegaDetector over a single image file.

    Args
    - im_file: str, path to image file
    - detector: loaded model
    - confidence_threshold: float, only detections above this threshold are returned
    - image: previously-loaded image, if available

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    
    if not quiet:
        print('Processing image {}'.format(im_file))
    
    if image is None:
        try:
            image = viz_utils.load_image(im_file)
        except Exception as e:
            if not quiet:
                print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': run_detector.FAILURE_IMAGE_OPEN
            }
            return result

    try:
        result = detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold, image_size=image_size)
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': run_detector.FAILURE_INFER
        }
        return result

    return result


#%% Main function

def load_and_run_detector_batch(model_file, image_file_names, checkpoint_path=None,
                                confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                                checkpoint_frequency=-1, results=None, n_cores=1,
                                use_image_queue=False, quiet=False, image_size=None):
    """
    Args
    - model_file: str, path to .pb model file
    - image_file_names: list of strings (image filenames), a single image filename, 
                        a folder to recursively search for images in, or a .json file containing
                        a list of images.
    - checkpoint_path: str, path to JSON checkpoint file
    - confidence_threshold: float, only detections above this threshold are returned
    - checkpoint_frequency: int, write results to JSON checkpoint file every N images
    - results: list of dict, existing results loaded from checkpoint
    - n_cores: int, # of CPU cores to use

    Returns
    - results: list of dict, each dict represents detections on one image
    """
    
    if n_cores is None:
        n_cores = 1
    
    if confidence_threshold is None:
        confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
        
    if checkpoint_frequency is None:
        checkpoint_frequency = -1
        
    # Handle the case where image_file_names is not yet actually a list
    if isinstance(image_file_names,str):
        
        # Find the images to score; images can be a directory, may need to recurse
        if os.path.isdir(image_file_names):
            image_dir = image_file_names
            image_file_names = ImagePathUtils.find_images(image_dir, True)
            print('{} image files found in folder {}'.format(len(image_file_names),image_dir))
            
        # A json list of image paths
        elif os.path.isfile(image_file_names) and image_file_names.endswith('.json'):
            list_file = image_file_names
            with open(list_file) as f:
                image_file_names = json.load(f)
            print('Loaded {} image filenames from list file {}'.format(len(image_file_names),list_file))
            
        # A single image file
        elif os.path.isfile(image_file_names) and ImagePathUtils.is_image_file(image_file_names):
            image_file_names = [image_file_names]
            print('Processing image {}'.format(image_file_names[0]))
            
        else:        
            raise ValueError('image_file_names is a string, but is not a directory, a json ' + \
                             'list (.json), or an image file (png/jpg/jpeg/gif)')
    
    if results is None:
        results = []

    already_processed = set([i['file'] for i in results])

    print('GPU available: {}'.format(is_gpu_available(model_file)))
    
    if n_cores > 1 and is_gpu_available(model_file):
        print('Warning: multiple cores requested, but a GPU is available; parallelization across ' + \
              'GPUs is not currently supported, defaulting to one GPU')
        n_cores = 1

    if n_cores > 1 and use_image_queue:
        print('Warning: multiple cores requested, but the image queue is enabled; parallelization ' + \
              'with the image queue is not currently supported, defaulting to one worker')
        n_cores = 1
        
    if use_image_queue:
        
        assert n_cores <= 1
        results = run_detector_with_image_queue(image_file_names, model_file, 
                                                confidence_threshold, quiet, 
                                                image_size=image_size)
        
    elif n_cores <= 1:

        # Load the detector
        start_time = time.time()
        detector = load_detector(model_file)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

        # Does not count those already processed
        count = 0

        for im_file in tqdm(image_file_names):

            # Will not add additional entries not in the starter checkpoint
            if im_file in already_processed:
                if not quiet:
                    print('Bypassing image {}'.format(im_file))
                continue

            count += 1

            result = process_image(im_file, detector, 
                                   confidence_threshold, quiet=quiet, 
                                   image_size=image_size)
            results.append(result)

            # Write a checkpoint if necessary
            if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
                
                print('Writing a new checkpoint after having processed {} images since '
                      'last restart'.format(count))
                
                assert checkpoint_path is not None                
                
                # Back up any previous checkpoints, to protect against crashes while we're writing
                # the checkpoint file.
                checkpoint_tmp_path = None
                if os.path.isfile(checkpoint_path):
                    checkpoint_tmp_path = checkpoint_path + '_tmp'
                    shutil.copyfile(checkpoint_path,checkpoint_tmp_path)
                    
                # Write the new checkpoint
                with open(checkpoint_path, 'w') as f:
                    json.dump({'images': results}, f, indent=1)
                    
                # Remove the backup checkpoint if it exists
                if checkpoint_tmp_path is not None:
                    os.remove(checkpoint_tmp_path)
                    
            # ...if it's time to make a checkpoint
            
    else:
        
        # When using multiprocessing, let the workers load the model
        detector = model_file

        print('Creating pool with {} cores'.format(n_cores))

        if len(already_processed) > 0:
            print('Warning: when using multiprocessing, all images are reprocessed')

        pool = workerpool(n_cores)

        image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
        results = pool.map(partial(process_images, detector=detector,
                                   confidence_threshold=confidence_threshold,image_size=image_size), 
                           image_batches)

        results = list(itertools.chain.from_iterable(results))

    # Results may have been modified in place, but we also return it for
    # backwards-compatibility.
    return results


def write_results_to_file(results, output_file, relative_path_base=None, 
                          detector_file=None, info=None, include_max_conf=False):
    """
    Writes list of detection results to JSON output file. Format matches:

    https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

    Args
    - results: list of dict, each dict represents detections on one image
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    # The typical case: we need to build the 'info' struct
    if info is None:
        
        info = { 
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.3' 
        }
        
        if detector_file is not None:
            detector_filename = os.path.basename(detector_file)
            detector_version = get_detector_version_from_filename(detector_filename)
            detector_metadata = get_detector_metadata_from_version_string(detector_version)
            info['detector'] = detector_filename  
            info['detector_metadata'] = detector_metadata
        else:
            info['detector'] = 'unknown'
            info['detector_metadata'] = get_detector_metadata_from_version_string('unknown')
        
    # If the caller supplied the entire "info" struct
    else:
        
        if detector_file is not None:
            
            print('Warning (write_results_to_file): info struct and detector file ' + \
                  'supplied, ignoring detector file')

    # The 'max_detection_conf' field used to be included by default, and it caused all kinds
    # of headaches, so it's no longer included unless the user explicitly requests it.
    if not include_max_conf:
        for im in results:
            if 'max_detection_conf' in im:
                del im['max_detection_conf']
            
    final_output = {
        'images': results,
        'detection_categories': run_detector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': info
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))


#%% Interactive driver

if False:
    
    pass

    #%%
    
    checkpoint_path = None
    model_file = r'G:\temp\models\md_v4.1.0.pb'
    confidence_threshold = 0.1
    checkpoint_frequency = -1
    results = None
    ncores = 1
    use_image_queue = False
    quiet = False
    image_dir = r'G:\temp\demo_images\ssmini'
    image_size = None
    image_file_names = ImagePathUtils.find_images(image_dir, recursive=False)    
    
    start_time = time.time()
    
    results = load_and_run_detector_batch(model_file=model_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=confidence_threshold,
                                          checkpoint_frequency=checkpoint_frequency,
                                          results=results,
                                          n_cores=ncores,
                                          use_image_queue=use_image_queue,
                                          quiet=quiet,
                                          image_size=image_size)
    
    elapsed = time.time() - start_time
    
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

    
#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Module to run a TF/PT animal detection model on lots of images')
    parser.add_argument(
        'detector_file',
        help='Path to detector model file (.pb or .pt)')
    parser.add_argument(
        'image_file',
        help='Path to a single image file, a JSON file containing a list of paths to images, or a directory')
    parser.add_argument(
        'output_file',
        help='Path to output JSON results file, should end with a .json extension')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--output_relative_filenames',
        action='store_true',
        help='Output relative file names, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--include_max_conf',
        action='store_true',
        help='Include the "max_detection_conf" field in the output')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-image console output')
    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help=('Force image resizing to a (square) integer size (not recommended to change this)'))    
    parser.add_argument(
        '--use_image_queue',
        action='store_true',
        help='Pre-load images, may help keep your GPU busy; does not currently support ' + \
             'checkpointing.  Useful if you have a very fast GPU and a very slow disk.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.0, don't include boxes below this " + \
            "confidence in the output file. Default is {}".format(
                run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD))
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=-1,
        help='Write results to a temporary file every N images; default is -1, which ' + \
             'disables this feature')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='File name to which checkpoints will be written if checkpoint_frequency is > 0')    
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to a JSON checkpoint file to resume from')
    parser.add_argument(
        '--allow_checkpoint_overwrite',
        action='store_true',
        help='By default, this script will bail if the specified checkpoint file ' + \
              'already exists; this option allows it to overwrite existing checkpoints')
    parser.add_argument(
        '--ncores',
        type=int,
        default=0,
        help='Number of cores to use; only applies to CPU-based inference, ' + \
             'does not support checkpointing when ncores > 1')
    parser.add_argument(
        '--class_mapping_filename',
        type=str,
        default=None,
        help='Use a non-default class mapping, supplied in a .json file with a dictionary mapping' + \
            'int-strings to strings.  This will also disable the addition of "1" to all category ' + \
            'IDs, so your class mapping should start at zero.')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), \
        'detector file {} does not exist'.format(args.detector_file)
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'
    assert args.output_file.endswith('.json'), 'output_file specified needs to end with .json'
    if args.checkpoint_frequency != -1:
        assert args.checkpoint_frequency > 0, 'Checkpoint_frequency needs to be > 0 or == -1'
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), \
            f'Could not find folder {args.image_file}, must supply a folder when ' + \
                '--output_relative_filenames is set'

    if os.path.exists(args.output_file):
        print('Warning: output_file {} already exists and will be overwritten'.format(
            args.output_file))

    # This is an experimental hack to allow the use of non-MD YOLOv5 models through
    # the same infrastructure; it disables the code that enforces MDv5-like class lists.
    if args.class_mapping_filename is not None:
        run_detector.USE_MODEL_NATIVE_CLASSES = True
        with open(args.class_mapping_filename,'r') as f:
            class_mapping = json.load(f)
        print('Loaded custom class mapping:')
        print(class_mapping)
        run_detector.DEFAULT_DETECTOR_LABEL_MAP = class_mapping
        
    # Load the checkpoint if available
    #
    # Relative file names are only output at the end; all file paths in the checkpoint are
    # still full paths.
    if args.resume_from_checkpoint is not None:
        assert os.path.exists(args.resume_from_checkpoint), \
            'File at resume_from_checkpoint specified does not exist'
        with open(args.resume_from_checkpoint) as f:
            print('Loading previous results from checkpoint file {}'.format(
                args.resume_from_checkpoint))
            saved = json.load(f)
        assert 'images' in saved, \
            'The checkpoint file does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []

    # Find the images to score; images can be a directory, may need to recurse
    if os.path.isdir(args.image_file):
        image_file_names = ImagePathUtils.find_images(args.image_file, args.recursive)
        if len(image_file_names) > 0:
            print('{} image files found in the input directory'.format(len(image_file_names)))
        else:
            if (args.recursive):
                print('No image files found in directory {}, exiting'.format(args.image_file))
            else:
                print('No image files found in directory {}, did you mean to specify '
                      '--recursive?'.format(
                    args.image_file))
            return
        
    # A json list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.json'):
        with open(args.image_file) as f:
            image_file_names = json.load(f)
        print('Loaded {} image filenames from list file {}'.format(
            len(image_file_names),args.image_file))
        
    # A single image file
    elif os.path.isfile(args.image_file) and ImagePathUtils.is_image_file(args.image_file):
        image_file_names = [args.image_file]
        print('Processing image {}'.format(args.image_file))
        
    else:        
        raise ValueError('image_file specified is not a directory, a json list, or an image file, '
                         '(or does not have recognizable extensions).')

    assert len(image_file_names) > 0, 'Specified image_file does not point to valid image files'
    assert os.path.exists(image_file_names[0]), \
        'The first image to be scored does not exist at {}'.format(image_file_names[0])

    output_dir = os.path.dirname(args.output_file)

    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
        
    assert not os.path.isdir(args.output_file), 'Specified output file is a directory'

    # Test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        
        if args.checkpoint_path is not None:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = os.path.join(output_dir,
                                           'checkpoint_{}.json'.format(
                                               datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        
        # Don't overwrite existing checkpoint files, this is a sure-fire way to eventually
        # erase someone's checkpoint.
        if (checkpoint_path is not None) and (not args.allow_checkpoint_overwrite) \
            and (args.resume_from_checkpoint is None):
            
            assert not os.path.isfile(checkpoint_path), \
                f'Checkpoint path {checkpoint_path} already exists, delete or move it before ' + \
                're-using the same checkpoint path, or specify --allow_checkpoint_overwrite'

        # Commenting this out for now... the scenario where we are resuming from a checkpoint,
        # then immediately overwrite that checkpoint with empty data is higher-risk than the 
        # annoyance of crashing a few minutes after starting a job.
        if False:
            # Confirm that we can write to the checkpoint path; this avoids issues where
            # we crash after several thousand images.
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': []}, f)
                
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
        
    else:
        
        checkpoint_path = None

    start_time = time.time()

    results = load_and_run_detector_batch(model_file=args.detector_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=args.threshold,
                                          checkpoint_frequency=args.checkpoint_frequency,
                                          results=results,
                                          n_cores=args.ncores,
                                          use_image_queue=args.use_image_queue,
                                          quiet=args.quiet,
                                          image_size=args.image_size)

    elapsed = time.time() - start_time
    images_per_second = len(results) / elapsed
    print('Finished inference for {} images in {} ({:.2f} images per second)'.format(
        len(results),humanfriendly.format_timespan(elapsed),images_per_second))

    relative_path_base = None
    if args.output_relative_filenames:
        relative_path_base = args.image_file
    write_results_to_file(results, args.output_file, relative_path_base=relative_path_base,
                          detector_file=args.detector_file,include_max_conf=args.include_max_conf)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
        print('Deleted checkpoint file {}'.format(checkpoint_path))

    print('Done!')


if __name__ == '__main__':
    main()
