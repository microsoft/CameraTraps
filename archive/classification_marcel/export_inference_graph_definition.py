#
# Creates a graph description, which is required to create a frozen graph.
# Adapted from from ./tf-slim/export_inference_graph.py
# Added preprocessing to the definition for easier handling
#


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

#tf.app.flags.DEFINE_integer(
#    'batch_size', None,
#    'Batch size for the exported model. Defaulted to "None" so batch size can '
#    'be specified at model runtime.')

tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('write_text_graphdef', False,
                         'Whether to write a text version of graphdef.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=FLAGS.is_training)

    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    input_shape = [None, None, 3]
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=input_shape)
    input_preprocessed = image_preprocessing_fn(placeholder, image_size, image_size)
    input_preprocessed = tf.expand_dims(input_preprocessed, 0)
    predictions = network_fn(input_preprocessed)[1]['Predictions']
    tf.identity(predictions, name='output')

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    graph_def = graph.as_graph_def()
    if FLAGS.write_text_graphdef:
      tf.io.write_graph(
          graph_def,
          os.path.dirname(FLAGS.output_file),
          os.path.basename(FLAGS.output_file),
          as_text=True)
    else:
      with gfile.GFile(FLAGS.output_file, 'wb') as f:
        f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
