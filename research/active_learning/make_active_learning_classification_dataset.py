'''
make_active_learning_dataset.py

Creates crops from detections in camera trap images for use in active learning for classification.

'''

import argparse, json, os, pickle, random, sys, tqdm, uuid
import numpy as np
import matplotlib; matplotlib.use('Agg')
from pycocotools.coco import COCO
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../../tfrecords/utils'))
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
