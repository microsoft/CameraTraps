DATASET_DIR=/data/lila/caltech/cct_cropped_tfrecords
TRAIN_DIR=./log/$(date +"%Y-%m-%d_%H.%M.%S")-logits
#TRAIN_DIR=./log/2019-02-14_11.27.03-mobilenet/
CHECKPOINT_PATH=/home/loris/git/tf-classification/pre-trained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt
MODEL_NAME=mobilenet_v2_140
CHECKPOINT_EXCLUDE=MobilenetV2/Logits
NUM_GPUS=1

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cct \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE} \
    --trainable_scopes=${CHECKPOINT_EXCLUDE} \
    --max_number_of_steps=20000 \
    --learning_rate=0.045 \
    --label_smoothing=0.1 \
    --batch_size=96 \
    --learning_rate_decay_factor=0.9 \
    --num_epochs_per_decay=1 \
    --moving_average_decay=0.9999

# Run evaluation.
python eval_image_classifier.py \
    --eval_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cct \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/all_lower_lr \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cct \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}/all \
    --learning_rate=0.0045 \
    --label_smoothing=0.1 \
    --batch_size=96 \
    --learning_rate_decay_factor=0.9 \
    --num_epochs_per_decay=5 \
    --moving_average_decay=0.9999

