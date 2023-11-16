DATASET_DIR=/data/lila/caltech/cct_cropped_tfrecords
TRAIN_DIR=./log/2019-02-14_11.27.03-mobilenet/all_lower_lr/
MODEL_NAME=mobilenet_v2_140
NUM_GPUS=1


# Run evaluation.
python eval_image_classifier.py \
    --eval_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cct \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}
