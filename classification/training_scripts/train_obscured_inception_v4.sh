DATASET_NAME=obscured
DATASET_DIR=/data2/idfg/obscured_dataset/tfrecords
#TRAIN_DIR=./log/$(date +"%Y-%m-%d_%H.%M.%S")_obscured_incv4
TRAIN_DIR=./log/2019-07-15_06.58.18_obscured_incv4
CHECKPOINT_PATH=/home/meerkat/git/tf-classification/pre-trained/inception_v4/inception_v4.ckpt
MODEL_NAME=inception_v4
CHECKPOINT_EXCLUDE=InceptionV4/AuxLogits,InceptionV4/Logits
NUM_GPUS=1

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/init \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE} \
    --trainable_scopes=${CHECKPOINT_EXCLUDE} \
    --max_number_of_steps=1000 \
    --batch_size=32 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=600 \
    --save_summaries_secs=600 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
    --eval_dir=${TRAIN_DIR}/init \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}/init

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/all \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}/init \
    --max_number_of_steps=1600 \
    --batch_size=32 \
    --learning_rate=0.00045 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=1000 \
    --save_interval_secs=180 \
    --save_summaries_secs=600 \
    --log_every_n_steps=10 \
    --optimizer=rmsprop \
    --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
    --eval_dir=${TRAIN_DIR}/all \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}/all
