DATASET_DIR=/data/lila/caltech/cct_cropped_tfrecords
TRAIN_DIR=./log/$(date +"%Y-%m-%d_%H.%M.%S")-logits
CHECKPOINT_PATH=/home/loris/git/tf-classification/pre-trained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cameratrap \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=Mobilenet/Logits \
    --trainable_scopes=Mobilenet/Logits
