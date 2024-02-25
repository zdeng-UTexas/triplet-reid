#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 PATH_TO_IMAGES RESNET_CHECKPOINT_FILE EXPERIMENT_ROOT ..."
    echo "See the README for more info"
    echo "Download ResNet-50 checkpoint from https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models"
    exit 1
fi

# Shift the arguments so that we can just forward the remainder.
IMAGE_ROOT=$1 ; shift # /home/dengzy/AEROPlan_Dataset
INIT_CHECKPT=$1 ; shift # /home/dengzy/AEROPlan_Dataset/Pretrained_Model/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt
EXP_ROOT=$1 ; shift # /home/dengzy/AEROPlan_Experiment/triplet-reid_EER


python3 train.py \
    --train_set /home/dengzy/AEROPlan_Dataset/DJI_0455_EER_images.csv \
    --model_name resnet_v1_50 \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --batch_p 5 \
    --batch_k 2 \
    --pre_crop_height 128 --pre_crop_width 128 \
    --net_input_height 128 --net_input_width 128 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 3e-5 \
    --train_iterations 250 \
    --decay_start_iteration 150 \
    "$@"

# python train.py \
#     --train_set /home/dengzy/AEROPlan_Dataset/DJI_0455_EER_images.csv \
#     --model_name resnet_v1_50 \ # Indicates the model architecture to be used for training, in this case, ResNet-50 version 1. ResNet-50 is a popular convolutional neural network CNN model used for image classification.
#     --image_root $IMAGE_ROOT \ # The root directory where the images specified in the training dataset are stored. IMAGE_ROOT is a variable that should be replaced with the actual path.
#     --initial_checkpoint $INIT_CHECKPT \ # Specifies the path to an initial checkpoint from which to start training. This allows for training to resume from a specific point or to start with a pre-trained model.
#     --experiment_root $EXP_ROOT \ # The directory where experiment outputs like model checkpoints and logs will be saved.
#     --flip_augment \ ## Enables data augmentation by flipping images horizontally.
#     --crop_augment \ # Enables random cropping of images as part of data augmentation.
#     --embedding_dim 128 \ # Sets the dimensionality of the embedding space to 128. This is relevant for models that output embeddings, such as those used for face recognition or person re-identification.
#     --batch_p 18 \ # The number of unique identities (persons) per batch. This is specific to triplet loss or similar training strategies where comparisons are made within and across identities.
#     --batch_k 4 \ # The number of images per identity in a batch. Used in conjunction with batch_p for training strategies that involve comparing multiple images of the same identity.
#     --pre_crop_height 128 --pre_crop_width 128 \ # Dimensions for cropping images before resizing them for model input. This is a form of data preprocessing.
#     --net_input_height 128 --net_input_width 128 \ # The dimensions to which images will be resized before being fed into the network. These dimensions must be compatible with the network architecture.
#     --margin soft \ # Specifies the use of a soft margin in the loss function, which can help in dealing with outliers or hard samples.
#     --metric euclidean \ # Uses the Euclidean distance as the metric for comparing embeddings in the loss function.
#     --loss batch_hard \ # Specifies the loss function to use, in this case, the batch hard loss which is commonly used in person re-identification tasks.
#     --learning_rate 3e-4 \ # Sets the initial learning rate for the optimizer. This is a key hyperparameter in training neural networks.
#     --train_iterations 25000 \ # The number of iterations (steps) to train the model.
#     --decay_start_iteration 15000 \ # Specifies when to start decaying the learning rate, allowing for more fine-grained updates to the model as training progresses.
#     "$@"
