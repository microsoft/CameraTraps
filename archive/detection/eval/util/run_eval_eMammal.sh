#!/bin/bash

EVAL_DIR=/yasiyu/mnt/emammaltrain/afs_out/ssdlite_mobilenetv2_coco_weights_loc_2class_24457_eval/

DETECTION_TFRECORD_FILE=$EVAL_DIR/eMammal_20180929_val.record

DETECTION_DICT_FILE=$EVAL_DIR/eMammal_20180929_val.json

TF_RECORD_FILES=$(ls -1 /yasiyu/mnt/emammaltrain/local_tf20180929/val* | tr '\n' ',')

python /yasiyu/tfodapi/tf_models/research/object_detection/inference/infer_detections.py --input_tfrecord_paths=$TF_RECORD_FILES --output_tfrecord_path=$DETECTION_TFRECORD_FILE --inference_graph=/yasiyu/mnt/emammaltrain/afs_out/ssdlite_mobilenetv2_coco_weights_loc_2class_24457/frozen_inference_graph.pb --discard_image_pixels

python tfrecords/read_from_tfrecords.py --input_tfrecord_file $DETECTION_TFRECORD_FILE --output_file $DETECTON_DICT_FILE
