DATASET_NAME=nacti
DATASET_DIR=/data/lila/nacti/cropped_tfrecords
TRAIN_DIR=./log/2019-03-12_09.10.31_nacti_incv4/
#TRAIN_DIR=log/2019-03-04_05.19.49_nacti_incv4
MODEL_NAME=inception_v4


# Run evaluation.
python eval_image_classifier.py \
    --eval_dir=${TRAIN_DIR}/all \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${TRAIN_DIR}/all
