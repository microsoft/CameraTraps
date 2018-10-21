import tensorflow as tf
import numpy as np
import iterate_tfrecords
import json
import pickle
import argparse


def read_from_tfrecords(data_path, features_to_extract):

    records = iterate_tfrecords.yield_record([data_path],features_to_extract)

    data = []
    for record in records:
        data.append(record)


    images = []
    #annotations = []
    boxes = []
    scores = []
    gts = []
    gt_labels = []
    box_labels = []
    for im in data:
        images.append(im['id'])
        h = im['height']
        w = im['width']
        im_boxes = []
        im_scores = []
        im_gts = []
        im_gt_labels = []
        im_box_labels = []
        for idx in range(len(im['detection_label'])):
            box = [im['detection_xmin'][idx]*w,im['detection_ymin'][idx]*h, im['detection_xmax'][idx]*w,im['detection_ymax'][idx]*h]
            im_boxes.append(box)
        im_scores=im['detection_score']
        im_box_labels=im['detection_label']
        for idx in range(len(im['label'])):
            box = [im['xmin'][idx]*w,im['ymin'][idx]*h, im['xmax'][idx]*w,im['ymax'][idx]*h]
            im_gts.append(box)
        im_gt_labels = im['label']
        boxes.append(im_boxes)
        scores.append(im_scores)
        gts.append(im_gts)
        gt_labels.append(im_gt_labels)
        box_labels.append(im_box_labels)

    detection_results = {}
    detection_results['images'] = images
    detection_results['detections'] = boxes
    detection_results['detection_scores'] = scores
    detection_results['gts'] = gts
    detection_results['gt_labels'] = gt_labels
    detection_results['detection_labels'] = box_labels
    
    return detection_results

def parse_args():

    parser = argparse.ArgumentParser(description = 'Make tfrecords from a CCT style json file')

    parser.add_argument('--input_tfrecord_file', dest='input_tfrecord_file',
                         help='Path to detection tfrecords',
                         type=str, required=True)
    parser.add_argument('--output_file', dest='output_file',
                         help='Path to store output dict',
                         type=str, required=True)
    parser.add_argument('--no_gt_bboxes', dest='no_gt_bboxes',
                         help='Flag to use if your tfrecords do not contain gtb bboxes',
                         action='store_true', default=False)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    if args.no_gt_bboxes:
        features_to_extract = [('image/id','id'),
                       ('image/class/label','label')
                       ('image/detection/label','detection_label'),
                       ('image/detection/bbox/xmin','detection_xmin'),
                       ('image/detection/bbox/xmax','detection_xmax'),
                       ('image/detection/bbox/ymin','detection_ymin'),
                       ('image/detection/bbox/ymax','detection_ymax'),
                       ('image/detection/score','detection_score'),
                       ('image/height','height'),
                       ('image/width', 'width')
                       ]
    else:
        features_to_extract = [('image/id','id'),
                       ('image/object/bbox/xmin','xmin'),
                       ('image/object/bbox/xmax','xmax'),
                       ('image/object/bbox/ymin','ymin'),
                       ('image/object/bbox/ymax','ymax'),
                       ('image/object/bbox/label','label'),
                       ('image/detection/label','detection_label'),
                       ('image/detection/bbox/xmin','detection_xmin'),
                       ('image/detection/bbox/xmax','detection_xmax'),
                       ('image/detection/bbox/ymin','detection_ymin'),
                       ('image/detection/bbox/ymax','detection_ymax'),
                       ('image/detection/score','detection_score'),
                       ('image/height','height'),
                       ('image/width', 'width')
                       ]

    detection_results = read_from_tfrecords(args.input_tfrecord_file, features_to_extract)
    pickle.dump(detection_results, open(args.output_file,'wb'))

if __name__ == '__main__':
    main()



