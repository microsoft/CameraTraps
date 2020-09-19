DATASET_DIR=/data/lila/caltech/cct_cropped_tfrecords
#TRAIN_DIR=./log/$(date +"%Y-%m-%d_%H.%M.%S")-logits
TRAIN_DIR=./log/$(ls log | tail -n 1)/
CHECKPOINT_PATH=/home/loris/git/tf-classification/pre-trained/inception_v4/inception_v4.ckpt
MODEL_NAME=inception_v4
NUM_GPUS=1

# Run evaluation.
python eval_image_classifier.py \
    --eval_dir=${TRAIN_DIR}/all \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cct \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}/all
