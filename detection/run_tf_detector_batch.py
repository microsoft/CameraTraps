"""
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

Sample invocation:

```
python run_tf_detector_batch.py "d:\temp\models\megadetector_v3.pb" "d:\temp\test_images" "d:\temp\test\out.json" --recursive
```

"""

#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import humanfriendly
from tqdm import tqdm

from detection.run_tf_detector import ImagePathUtils, TFDetector
import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())


#%% Main function

def load_and_run_detector_batch(model_file, image_file_names, checkpoint_path,
                                confidence_threshold, checkpoint_frequency, results):
    already_processed = set([i['file'] for i in results])

    # load the detector
    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    count = 0  # does not count those already processed
    for im_file in tqdm(image_file_names):
        if im_file in already_processed:  # will not add additional entries not in the starter checkpoint
            continue

        count += 1

        try:
            image = viz_utils.load_image(im_file)
        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            results.append(result)
            continue

        try:
            result = tf_detector.generate_detections_one_image(image, im_file, detection_threshold=confidence_threshold)
            results.append(result)

        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(im_file, e))
            continue

        # checkpoint
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count))
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': results}, f)

    return results  # actually modified in place


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(
        description='Module to run a TF animal detection model on lots of images'
    )
    parser.add_argument(
        'detector_file',
        help='Path to .pb TensorFlow detector model file'
    )
    parser.add_argument(
        'image_file',
        help='Can be a single image file, a json file containing a list of paths to images, or a directory'
    )
    parser.add_argument(
        'output_file',
        help='Output results file, should end with a .json extension')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if --image_file points to a directory')
    parser.add_argument(
        '--output_relative_filenames',
        action='store_true',
        help='Output relative file names, only meaningful if --image_file points to a directory')
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
        help='Initiate from the specified checkpoint, which is in the same directory as the output_file specified')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'detector_file specified does not exist'
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison
    assert args.output_file.endswith('.json'), 'output_file specified needs to end with .json'
    if args.checkpoint_frequency != -1:
        assert args.checkpoint_frequency > 0, 'Checkpoint_frequency needs to be > 0 or == -1'
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), 'Since output_relative_filenames is flagged, image_file needs to be a directory'

    if os.path.exists(args.output_file):
        print('Warning: output_file {} already exists and will be overwritten'.format(args.output_file))

    # load the checkpoint if available
    # relative file names are only output at the end; all file paths in the checkpoint are still full paths
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

    # find the images to score; images can be:
    # a directory, may need to recurse
    if os.path.isdir(args.image_file):
        image_file_names = ImagePathUtils.find_images(args.image_file, args.recursive)
        print('{} image files found in the input directory'.format(len(image_file_names)))
    # a json list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.json'):
        with open(args.image_file) as f:
            image_file_names = json.load(f)
        print('{} image files found in the json list'.format(len(image_file_names)))
    # a single image file
    elif os.path.isfile(args.image_file) and ImagePathUtils.is_image_file(args.image_file):
        image_file_names = [args.image_file]
        print('A single image at {} is the input file'.format(args.image_file))
    else:
        print('image_file specified is not a directory, a json list or an image file (or does not have recognizable extensions), exiting.')
        sys.exit(1)

    assert len(image_file_names) > 0, 'image_file provided does not point to valid image files'
    assert os.path.exists(image_file_names[0]), 'The first image to be scored does not exist at {}'.format(image_file_names[0])

    # test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        output_dir = os.path.dirname(args.output_file)
        checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': []}, f)
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
    else:
        checkpoint_path = None

    results = load_and_run_detector_batch(model_file=args.detector_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=args.threshold,
                                          checkpoint_frequency=args.checkpoint_frequency,
                                          results=results)

    if args.output_relative_filenames:
        for r in results:
            r['file'] = os.path.relpath(r['file'], start=args.image_file)

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(args.output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(args.output_file))

    # finally delete the checkpoint file if used
    if checkpoint_path:
        os.remove(checkpoint_path)
        print('Deleted checkpoint file')
    print('Done!')


if __name__ == '__main__':
    main()
