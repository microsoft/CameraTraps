'''
make_active_learning_dataset.py

Creates crops from detections in camera trap images for use in active learning for classification.
Largely drawn from CameraTraps/data_management/databases/classification/make_classification_dataset.py.

'''

import argparse, cv2, glob, json, os, pickle, random, sys, tqdm, uuid
import numpy as np
import matplotlib; matplotlib.use('Agg')
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../../data_management/tfrecords/utils'))
if sys.version_info.major >= 3:
  import create_tfrecords_py3 as tfr
else:
  import create_tfrecords as tfr

print('If you run into import errors, please make sure you added "models/research" and ' +\
      ' "models/research/object_detection" of the tensorflow models repo to the PYTHONPATH\n\n')
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


########################################################## 
### Configuration

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', type=str, default='missouricameratraps/lila',
                    help='Root folder of the images.')
parser.add_argument('output_dir', type=str, default='missouricameratraps/crops',
                    help='Output folder for cropped images, used as inputs for classification model.')
parser.add_argument('frozen_graph', type=str, default='frozen_inference_graph.pb',
                    help='Frozen graph of detection network as created by export_inference_graph.py of TFODAPI.')
parser.add_argument('--detection_confidence', type=float, default=0.9,
                    help='Confidence threshold for deciding which detections to keep.')
parser.add_argument('--padding_factor', type=float, default=1.3*1.3,
                    help='We will crop a tight square box around the animal enlarged by this factor. ' + \
                    'Default is 1.3 * 1.3 = 1.69, which accounts for the cropping at test time and for' + \
                    ' a reasonable amount of context')
args = parser.parse_args()


##########################################################
### The actual code

# Check arguments
IMAGE_DIR = args.image_dir
assert os.path.exists(IMAGE_DIR), IMAGE_DIR + ' does not exist'

OUTPUT_DIR = args.output_dir
# Create output directories
if not os.path.exists(OUTPUT_DIR):
    print('Creating crops output directory.')
    os.makedirs(OUTPUT_DIR)

PATH_TO_FROZEN_GRAPH = args.frozen_graph
assert os.path.isfile(PATH_TO_FROZEN_GRAPH), PATH_TO_FROZEN_GRAPH + ' does not exist'

# Padding around the detected objects when cropping
# 1.3 for the cropping during test time and 1.3 for 
# the context that the CNN requires in the left-over 
# image
PADDING_FACTOR = args.padding_factor
assert PADDING_FACTOR >= 1, 'Padding factor should be equal or larger 1'

DETECTION_CONFIDENCE = args.detection_confidence


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
graph = detection_graph


detections = dict()
crop_info_json = []
with graph.as_default():
    with tf.Session() as sess:
        ### Preparations: get all the output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boigxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # For all images in the image directoryig
        imgs_in_dir = glob.glob(os.path.join(IMAGE_DIR, '**/*.JPG'), recursive=True) # All images in directory (may be a subset of the dataset)
        for cur_image in tqdm.tqdm(sorted(imgs_in_dir)):
            
            # Load image
            image = np.array(Image.open(cur_image))
            if image.dtype != np.uint8:
                print('Failed to load image ' + cur_image)
                continue

            # Run inference
            output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            # Add detections to the collection
            detections[cur_image] = output_dict
            
            # Get info about the image
            imsize = Image.open(cur_image).size
            imwidth = imsize[0]
            imheight = imsize[1]
            img_cv2 = cv2.imread(cur_image, cv2.IMREAD_COLOR)
            channel0_mean = np.mean(img_cv2[:,:,0])
            channel1_mean = np.mean(img_cv2[:,:,1])
            channel2_mean = np.mean(imimageg_cv2[:,:,2])
            max_channel_mean = max(channel0_mean, channel1_mean, channel2_mean)
            min_channel_mean = min(channel0_mean, channel1_mean, channel2_mean)
            if (max_channel_mean - min_channel_mean) < 1:
                grayscale = True
            else:
                grayscale = False
            # Select detections with a confidence larger than DETECTION_CONFIDENCE
            selection = output_dict['detection_scores'] > DETECTION_CONFIDENCE
            # Get these boxes and convert normalized coordinates to pixel coordinates
            selected_boxes = (output_dict['detection_boxes'][selection] * np.tile([imsize[1],imsize[0]], (1,2)))
            # Pad the detected animal to a square box and additionally by PADDING_FACTOR, the result will be in crop_boxes
            # However, we need to make sure that it box coordinates are still within the image
            bbox_sizes = np.vstack([selected_boxes[:,2] - selected_boxes[:,0], selected_boxes[:,3] - selected_boxes[:,1]]).T
            offsets = (PADDING_FACTOR * np.max(bbox_sizes, axis=1, keepdims=True) - bbox_sizes) / 2
            crop_boxes = selected_boxes + np.hstack([-offsets,offsets])
            crop_boxes = np.maximum(0,crop_boxes).astype(int)

            # For each detected bounding box with high confidence, we will
            # crop the image to the padded box and save it
            for box_id in range(selected_boxes.shape[0]):
                crop_info = dict()

                # generate a unique identifier for the detection
                detection_id = str(uuid.uuid4())
                crop_info['id'] = detection_id
                crop_info['image'] = cur_image.split(IMAGE_DIR)[1]
                crop_info['grayscale'] = grayscale

                # bbox is the detected box, crop_box the padded / enlarged box
                bbox, crop_box = selected_boxes[box_id], crop_boxes[box_id]
                new_file_name = os.path.split(cur_image)[1]
                crop_box_area = (crop_box[1] - crop_box[0])*(crop_box[3] - crop_box[2])
                img_area = imwidth*imheight
                crop_info['relative_size'] = float(crop_box_area)/img_area

                # Add numbering to the original file name if there are multiple boxes
                if selected_boxes.shape[0] > 1:
                    new_file_base, new_file_ext = os.path.splitext(new_file_name)
                    new_file_name = '{}_{}{}'.format(new_file_base, box_id, new_file_ext)
                
                # The absolute file path where we will store the image
                out_file = os.path.join(OUTPUT_DIR, new_file_name)
                
                if not os.path.exists(out_file):
                    try:
                        img = np.array(Image.open(cur_image))
                        cropped_img = img[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3]]
                        Image.fromarray(cropped_img).save(out_file)
                    except ValueError:
                        continue
                    except FileNotFoundError:
                        continue
                else:
                    # if COCO_OUTPUT_DIR is set, then we will only use the shape
                    # of cropped_img in the following code. So instead of reading 
                    # cropped_img = np.array(Image.open(out_file))
                    # we can speed everything up by reading only the size of the image
                    cropped_img = np.zeros((3,) + Image.open(out_file).size).T

print('Crops created for images in '+IMAGE_DIR+' and stored in '+OUTPUT_DIR)