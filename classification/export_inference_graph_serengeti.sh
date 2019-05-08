DATASET_NAME=serengeti
DATASET_DIR=/data/lila/serengeti/serengeti_cropped_tfrecords
#CHECKPOINT_DIR=./$(echo log/20* | sort | tail -n 1)/all
CHECKPOINT_DIR=/home/loris/git/tf-detection/CameraTraps/classification/log/serengeti_cropped_resnext_86.3/all
MODEL_NAME=inception_v4
OUTPUT_NODE_NAMES=InceptionV4/Logits/Predictions
NUM_GPUS=1

python ../export_inference_graph_with_preprocessing.py \
    --model_name=${MODEL_NAME} \
    --output_file=${CHECKPOINT_DIR}/${MODEL_NAME}_inf_graph_def.pbtxt \
    --dataset_name=serengeti \
    --write_text_graphdef=True


python ../freeze_and_optimize_graph.py \
    --input_graph=${CHECKPOINT_DIR}/${MODEL_NAME}_inf_graph_def.pbtxt \
    --input_checkpoint=`ls ${CHECKPOINT_DIR}/model.ckpt*meta | tail -n 1 | rev | cut -c 6- | rev` \
    --output_graph=${CHECKPOINT_DIR}/frozen_inference_graph.pb \
    --input_node_names=input \
    --output_node_names=${OUTPUT_NODE_NAMES} \
    --clear_devices=True 
