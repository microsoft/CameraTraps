"""
Reference:
https://colab.research.google.com/drive/1NSXffwYbZfnnE4waooWCBFujcdGTZvhx?authuser=2#scrollTo=wh_HPMOqWH9z
https://stackoverflow.com/questions/58555159/set-multiple-tfrecord-into-config-file-of-tensorflow-object-detection-api

"""
import sys
root_path = f"/home/azureuser/cameratrap_efficient/CameraTraps/detection/efficient/tf2_workspace/"
sys.path.append(f'{root_path}')

model_variant = sys.argv[1]
"""
'd0', 'Model variant of efficientdet . Ex: d0, d1, d2, d3, ..., d7.')
"""

if model_variant is None:
    model_variant = 'd0'

import os
import numpy as np
from six import BytesIO
from object_detection.utils import config_util
import tensorflow as tf
import sys
# Since we are calling from elsewhere
sys.path.append(f'{root_path}/models')
sys.path.append(f'{root_path}/models/research')


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import colab_utils
from object_detection.builders import model_builder



"""
Note: Update these TFRecord names from "cells" and "cells_label_map" to your files!
"""
# Reference - https://stackoverflow.com/questions/58555159/set-multiple-tfrecord-into-config-file-of-tensorflow-object-detection-api
train_record_fname = "/camtrap/tfrecord/megadetectorv4-1/mdv4box01_train-?????-of-?????"
test_record_fname = "/camtrap/tfrecord/megadetectorv4-1/mdv4box01_val__-?????-of-?????"
# Reference - https://github.com/microsoft/CameraTraps/blob/master/detection/detector_training/experiments/megadetector_v4/label_map.pbtxt
label_map_pbtxt_fname = '/camtrap/tfrecord/camtrap_label_map.pbtxt'

##change chosen model to deploy different models available in the TF2 object detection zoo
MODELS_CONFIG = {
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 8
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 4
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 2
    },
        'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 1
    },
        'efficientdet-d4': {
        'model_name': 'efficientdet_d4_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d4_1024x1024_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d4_coco17_tpu-32.tar.gz',
        'batch_size': 1
    },
        'efficientdet-d5': {
        'model_name': 'efficientdet_d5_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d5_1280x1280_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d5_coco17_tpu-32.tar.gz',
        'batch_size': 1
    },
        'efficientdet-d6': {
        'model_name': 'efficientdet_d6_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d6_1408x1408_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d6_coco17_tpu-32.tar.gz',
        'batch_size': 1
    },
        'efficientdet-d7': {
        'model_name': 'efficientdet_d7_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d7_1536x1536_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d7_coco17_tpu-32.tar.gz',
        'batch_size': 1
    }
}

#in this tutorial we implement the lightweight, smallest state of the art efficientdet model
#if you want to scale up tot larger efficientdet models you will likely need more compute!
chosen_model = f'efficientdet-{model_variant}'

num_steps = 40000 #The more steps, the longer the training. Increase if your loss function is still decreasing and validation metrics are increasing. 
num_eval_steps = 500 #Perform evaluation after so many steps

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
batch_size = MODELS_CONFIG[chosen_model]['batch_size'] #if you can fit a large batch in memory, it may speed up your training


#download pretrained weights
modelspath =  f'{root_path}models/research/deploy/'
if not os.path.exists(modelspath): os.mkdir(modelspath)
download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint

if not os.path.exists(f'{modelspath}/{pretrained_checkpoint}'):
    import tarfile
    os.system(f'wget {download_tar} -O {modelspath}/{pretrained_checkpoint}')
    tar = tarfile.open(f'{modelspath}/{pretrained_checkpoint}')
    tar.extractall(path=f'{modelspath}')
    tar.close()

download_config = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/' + base_pipeline_file
if not os.path.exists(f'{modelspath}/{base_pipeline_file}'):
    os.system(f'wget {download_config} -O {modelspath}/{base_pipeline_file}')

#prepare
pipeline_fname = f'{modelspath}' + base_pipeline_file
fine_tune_checkpoint = f'{modelspath}' + model_name + '/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


num_classes = get_num_classes(label_map_pbtxt_fname)
print(num_classes)

#write custom configuration file by slotting our dataset, model checkpoint, and training parameters into the base pipeline file

import re


print('writing custom configuration file')

with open(pipeline_fname) as f:
    s = f.read()

with open(f'{modelspath}/pipeline_file.config', 'w') as f:    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    
    #fine-tune checkpoint type
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
        
    f.write(s)

pipeline_file = f'{modelspath}/pipeline_file.config'
model_dir = f'{root_path}/training/'

# python models/research/object_detection/model_main_tf2.py \
#     --pipeline_config_path= 'models/research/deploy/pipeline_file.config'\
#     --model_dir='training' \
#     --alsologtostderr \
#     --num_train_steps=40000 \
#     --sample_1_of_n_eval_examples=1 \
#     --num_eval_steps=500