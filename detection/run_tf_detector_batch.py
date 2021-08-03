r"""
Module to run a TensorFlow animal detection model on lots of images, writing the results
to a file in the same format produced by our batch API:

https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing

This enables the results to be used in our post-processing pipeline; see
api/batch_processing/postprocessing/postprocess_batch_results.py .

This script has *somewhat* tested functionality to save results to checkpoints
intermittently, in case disaster strikes. To enable this, set --checkpoint_frequency
to n > 0, and results will be saved as a checkpoint every n images. Checkpoints
will be written to a file in the same directory as the output_file, and after all images
are processed and final results file written to output_file,
the temporary checkpoint file will be deleted. If you want to resume from a checkpoint,
set the checkpoint file's path using --resume_from_checkpoint.

The `threshold` you can provide as an argument is the confidence threshold above which detections
will be included in the output file.

Has preliminary multiprocessing support for CPUs only; if a GPU is available, it will
use the GPU instead of CPUs, and the --ncores option will be ignored.  Checkpointing
is not supported when using multiprocessing.

Sample invocation:

python run_tf_detector_batch.py "d:\temp\models\md_v4.1.0.pb" "d:\temp\test_images" "d:\temp\out.json" --recursive
"""

#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly
from tqdm import tqdm
# from multiprocessing.pool import ThreadPool as workerpool
from multiprocessing.pool import Pool as workerpool

# Useful hack to force CPU inference
#
# Need to do this before any TF imports
force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from detection.run_tf_detector import ImagePathUtils, TFDetector
import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('tf.test.is_gpu_available:', tf.test.is_gpu_available())


#%% Support functions for multiprocessing

def process_images(im_files, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a list of image files.

    Args
    - im_files: list of str, paths to image files
    - tf_detector: TFDetector (loaded model) or str (path to .pb model file)
    - confidence_threshold: float, only detections above this threshold are returned

    Returns
    - results: list of dict, each dict represents detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    if isinstance(tf_detector, str):
        start_time = time.time()
        tf_detector = TFDetector(tf_detector)
        elapsed = time.time() - start_time
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    results = []
    for im_file in im_files:
        results.append(process_image(im_file, tf_detector, confidence_threshold))
    return results


def process_image(im_file, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a single image file.

    Args
    - im_file: str, path to image file
    - tf_detector: TFDetector, loaded model
    - confidence_threshold: float, only detections above this threshold are returned

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    print('Processing image {}'.format(im_file))
    try:
        image = viz_utils.load_image(im_file)
    except Exception as e:
        print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_IMAGE_OPEN
        }
        return result

    try:
        result = tf_detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold)
    except Exception as e:
        print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_TF_INFER
        }
        return result

    return result


def chunks_by_number_of_chunks(ls, n):
    """Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]


#%% Main function

def load_and_run_detector_batch(model_file, image_file_names, checkpoint_path=None,
                                confidence_threshold=0, checkpoint_frequency=-1,
                                results=None, n_cores=0):
    """
    Args
    - model_file: str, path to .pb model file
    - image_file_names: list of str, paths to image files
    - checkpoint_path: str, path to JSON checkpoint file
    - confidence_threshold: float, only detections above this threshold are returned
    - checkpoint_frequency: int, write results to JSON checkpoint file every N images
    - results: list of dict, existing results loaded from checkpoint
    - n_cores: int, # of CPU cores to use

    Returns
    - results: list of dict, each dict represents detections on one image
    """
    if results is None:
        results = []

    already_processed = set([i['file'] for i in results])

    if n_cores > 1 and tf.test.is_gpu_available():
        print('Warning: multiple cores requested, but a GPU is available; parallelization across GPUs is not currently supported, defaulting to one GPU')

    # If we're not using multiprocessing...
    if n_cores <= 1 or tf.test.is_gpu_available():

        # Load the detector
        start_time = time.time()
        tf_detector = TFDetector(model_file)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

        # Does not count those already processed
        count = 0

        for im_file in tqdm(image_file_names):

            # Will not add additional entries not in the starter checkpoint
            if im_file in already_processed:
                print('Bypassing image {}'.format(im_file))
                continue

            count += 1

            result = process_image(im_file, tf_detector, confidence_threshold)
            results.append(result)

            # checkpoint
            if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
                print('Writing a new checkpoint after having processed {} images since last restart'.format(count))
                with open(checkpoint_path, 'w') as f:
                    json.dump({'images': results}, f)

    else:
        # when using multiprocessing, let the workers load the model
        tf_detector = model_file

        print('Creating pool with {} cores'.format(n_cores))

        if len(already_processed) > 0:
            print('Warning: when using multiprocessing, all images are reprocessed')

        pool = workerpool(n_cores)

        image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
        results = pool.map(partial(process_images, tf_detector=tf_detector,
                                   confidence_threshold=confidence_threshold), image_batches)

        results = list(itertools.chain.from_iterable(results))

    # results may have been modified in place, but we also return it for backwards-compatibility.
    return results


def write_results_to_file(results, output_file, relative_path_base=None):
    """Writes list of detection results to JSON output file. Format matches
    https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

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

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Module to run a TF animal detection model on lots of images')
    parser.add_argument(
        'detector_file',
        help='Path to .pb TensorFlow detector model file')
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
        '--threshold',
        type=float,
        default=TFDetector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.0, don't include boxes below this confidence in the output file. Default is 0.1")
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=-1,
        help='Write results to a temporary file every N images; default is -1, which disables this feature')
    parser.add_argument(
        '--resume_from_checkpoint',
        help='Path to a JSON checkpoint file to resume from, must be in same directory as output_file')
    parser.add_argument(
        '--ncores',
        type=int,
        default=0,
        help='Number of cores to use; only applies to CPU-based inference, does not support checkpointing when ncores > 1')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'Specified detector_file does not exist'
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison
    assert args.output_file.endswith('.json'), 'output_file specified needs to end with .json'
    if args.checkpoint_frequency != -1:
        assert args.checkpoint_frequency > 0, 'Checkpoint_frequency needs to be > 0 or == -1'
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), 'image_file must be a directory when --output_relative_filenames is set'

    if os.path.exists(args.output_file):
        print('Warning: output_file {} already exists and will be overwritten'.format(args.output_file))

    # Load the checkpoint if available
    #
    # Relative file names are only output at the end; all file paths in the checkpoint are
    # still full paths.
    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'
        with open(args.resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'The file saved as checkpoint does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []

    # Find the images to score; images can be a directory, may need to recurse
    if os.path.isdir(args.image_file):
        image_file_names = ImagePathUtils.find_images(args.image_file, args.recursive)
        print('{} image files found in the input directory'.format(len(image_file_names)))
    # A json list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.json'):
        with open(args.image_file) as f:
            image_file_names = json.load(f)
        print('{} image files found in the json list'.format(len(image_file_names)))
    # A single image file
    elif os.path.isfile(args.image_file) and ImagePathUtils.is_image_file(args.image_file):
        image_file_names = [args.image_file]
        print('A single image at {} is the input file'.format(args.image_file))
    else:
        raise ValueError('image_file specified is not a directory, a json list, or an image file, '
                         '(or does not have recognizable extensions).')

    assert len(image_file_names) > 0, 'Specified image_file does not point to valid image files'
    assert os.path.exists(image_file_names[0]), 'The first image to be scored does not exist at {}'.format(image_file_names[0])

    output_dir = os.path.dirname(args.output_file)

    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
        
    assert not os.path.isdir(args.output_file), 'Specified output file is a directory'

    # Test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
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
                                          n_cores=args.ncores)

    elapsed = time.time() - start_time
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

    relative_path_base = None
    if args.output_relative_filenames:
        relative_path_base = args.image_file
    write_results_to_file(results, args.output_file, relative_path_base=relative_path_base)

    if checkpoint_path:
        os.remove(checkpoint_path)
        print('Deleted checkpoint file')

    print('Done!')


if __name__ == '__main__':
    main()
