DATASET_NAME=serengeti
# IMPORTANT: Please specify the path without trailing /
CHECKPOINT_DIR=./log/init_test/init
MODEL_NAME=inception_v4
NUM_GPUS=1

python ../export_inference_graph_definition.py \
    --model_name=${MODEL_NAME} \
    --output_file=${CHECKPOINT_DIR}/${MODEL_NAME}_inf_graph_def.pbtxt \
    --dataset_name=${DATASET_NAME} \
    --write_text_graphdef=True


python ../freeze_graph.py \
    --input_graph=${CHECKPOINT_DIR}/${MODEL_NAME}_inf_graph_def.pbtxt \
    --input_checkpoint=`ls ${CHECKPOINT_DIR}/model.ckpt*meta | tail -n 1 | rev | cut -c 6- | rev` \
    --output_graph=${CHECKPOINT_DIR}/frozen_inference_graph.pb \
    --input_node_names=input \
    --output_node_names=output \
    --clear_devices=True 
