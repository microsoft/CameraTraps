import argparse
import tensorflow as tf
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--frozen_graph', type=str,
                    help='Frozen graph of detection network as create by export_inference_graph.py of TFODAPI.')
parser.add_argument('--classlist', type=str,
                    help='Path to text file containing the names of all possible categories.')
parser.add_argument('--image_path', type=str,
                    help='Path to image file.')
args = parser.parse_args()

# Check that all files exists for easier debugging
assert os.path.exists(args.frozen_graph)
assert os.path.exists(args.classlist)
assert os.path.exists(args.image_path)

# Load frozen graph
model_graph = tf.Graph()
with model_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.frozen_graph, 'rb') as fid:
      od_graph_def.ParseFromString(fid.read())
      tf.import_graph_def(od_graph_def, name='')
graph = model_graph

# Load class list
classlist = open(args.classlist, 'rt').read().splitlines()
# Remove empty lines
classlist = [li for li in classlist if len(li)>0]

with model_graph.as_default():
    with tf.Session() as sess:
        # Collect tensors for input and output
        image_tensor = tf.get_default_graph().get_tensor_by_name('input:0')
        predictions_tensor = tf.get_default_graph().get_tensor_by_name('output:0')
        predictions_tensor = tf.squeeze(predictions_tensor, [0])

        # Read imag
        with open(args.image_path, 'rb') as fi:
            image = sess.run(tf.image.decode_jpeg(fi.read(), channels=3))
            image = image / 255.

        # Run inference
        predictions = sess.run(predictions_tensor, feed_dict={image_tensor: image})

        # Print output
        print('Prediction finished. Most likely classes:')
        for class_id in np.argsort(-predictions)[:5]:
            print('    "{}" with confidence {:.2f}%'.format(classlist[class_id], predictions[class_id]*100))
