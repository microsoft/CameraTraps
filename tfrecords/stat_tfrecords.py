"""
These utility functions are meant for computing basic statistics in a set of tfrecord
files. They can be used to sanity check the training and testing files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

def class_stats(tfrecords):
    """
    Sum the number of images and compute the number of images available for each class.
    """

    filename_queue = tf.train.string_input_producer(
        tfrecords,
        num_epochs=1
    )

    # Construct a Reader to read examples from the .tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/class/label' : tf.FixedLenFeature([], tf.int64)
        }
    )

    label = features['image/class/label']

    image_count = 0
    class_image_count = {}

    coord = tf.train.Coordinator()
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            while not coord.should_stop():

                outputs = sess.run([label])

                class_label = outputs[0]
                if class_label not in class_image_count:
                    class_image_count[class_label] = 0
                class_image_count[class_label] += 1
                image_count += 1


        except tf.errors.OutOfRangeError as e:
            pass

    # Basic info
    print("Found %d images" % (image_count,))
    print("Found %d classes" % (len(class_image_count),))

    class_labels = class_image_count.keys()
    class_labels.sort()

    # Print out the per class image counts
    print("Class Index | Image Count")
    for class_label in class_labels:
        print("{0:11d} | {1:6d} ".format(class_label, class_image_count[class_label]))

    if len(class_labels) == 0:
        return

    # Can we detect if there any missing classes?
    max_class_index = max(class_labels)

    # We expect class id for each value in the range [0, max_class_id]
    # So lets see if we are missing any of these values
    missing_values = list(set(range(max_class_index+1)).difference(class_labels))
    if len(missing_values) > 0:
        print("WARNING: expected %d classes but only found %d classes." %
              (max_class_index, len(class_labels)))
        missing_values.sort()
        for index in missing_values:
            print("Missing class %d" % (index,))

def verify_bboxes(tfrecords):

    filename_queue = tf.train.string_input_producer(
        tfrecords,
        num_epochs=1
    )

    # Construct a Reader to read examples from the .tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/id' : tf.FixedLenFeature([], tf.string),
            'image/height' : tf.FixedLenFeature([], tf.int64),
            'image/width' : tf.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/count' : tf.FixedLenFeature([], tf.int64)
        }
    )

    image_height = tf.cast(features['image/height'], tf.float32)
    image_width = tf.cast(features['image/width'], tf.float32)

    image_id = features['image/id']

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    num_bboxes = tf.cast(features['image/object/count'], tf.int32)

    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes, [1, 0])

    fetches = [image_id, image_height, image_width, bboxes, num_bboxes]

    image_count = 0
    bbox_widths = []
    bbox_heights = []
    images_with_small_bboxes = set()
    images_with_reversed_coords = set()
    images_with_bbox_count_mismatch = set()

    coord = tf.train.Coordinator()
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            while not coord.should_stop():

                outputs = sess.run(fetches)

                img_id = outputs[0]
                img_h = outputs[1]
                img_w = outputs[2]
                img_bboxes = outputs[3]
                img_num_bboxes = outputs[4]

                if img_bboxes.shape[0] != img_num_bboxes:
                    images_with_bbox_count_mismatch.add(img_id)

                for img_bbox in img_bboxes:
                    x1, y1, x2, y2 = img_bbox

                    # Reversed coordinates?
                    if x1 > x2:
                        images_with_reversed_coords.add(img_id)
                        t = x1
                        x1 = x2
                        x2 = t
                    if y1 > y2:
                        images_with_reversed_coords.add(img_id)
                        t = y1
                        y1 = y2
                        y2 = t

                    w = (x2 - x1) * img_w
                    h = (y2 - y1) * img_h

                    # Too small of an area?
                    if w * h < 10:
                        images_with_small_bboxes.add(img_id)

                    bbox_widths.append(w)
                    bbox_heights.append(h)

                image_count += 1


        except tf.errors.OutOfRangeError as e:
            pass

    # Basic info
    print("Found %d images" % (image_count,))
    print()
    print("Found %d images with small bboxes" % (len(images_with_small_bboxes),))
    #print("Images with areas < 10:")
    #for img_id in images_with_small_bboxes:
    #    print(img_id)
    print()
    print("Found %d images with reversed coordinates" %
          (len(images_with_reversed_coords),))
    #print("Images with reversed coordinates:")
    #for img_id in images_with_reversed_coords:
    #    print(img_id)
    print()
    print("Found %d images with bbox count mismatches" %
          (len(images_with_bbox_count_mismatch),))
    #for img_id in images_with_bbox_count_mismatch:
    #    print(img_id)
    print()

    bbox_widths = np.round(np.array(bbox_widths)).astype(int)
    bbox_heights = np.round(np.array(bbox_heights)).astype(int)

    print("Mean width: %0.4f" % (np.mean(bbox_widths),))
    print("Median width: %d" % (np.median(bbox_widths),))
    print("Max width: %d" % (np.max(bbox_widths),))
    print("Min width: %d" % (np.min(bbox_widths),))
    print()
    print("Mean height: %0.4f" % (np.mean(bbox_heights),))
    print("Median height: %d" % (np.median(bbox_heights),))
    print("Max height: %d" % (np.max(bbox_heights),))
    print("Min height: %d" % (np.min(bbox_heights),))


def parse_args():

    parser = argparse.ArgumentParser(description='Basic statistics on tfrecord files')

    parser.add_argument('--stat', dest='stat_type',
                        choices=['class_stats', 'verify_bboxes'],
                        required=True)

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)


    parsed_args = parser.parse_args()

    return parsed_args

def main():
    parsed_args = parse_args()

    if parsed_args.stat_type == 'class_stats':
        class_stats(parsed_args.tfrecords)
    elif parsed_args.stat_type == 'verify_bboxes':
        verify_bboxes(parsed_args.tfrecords)

if __name__ == '__main__':
    main()
