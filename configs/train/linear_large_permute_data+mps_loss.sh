#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=4096
USE_PERMUTABLE_SUBSEQ_LOSS=true
PERMUTE_MPS=true
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION=5

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=12
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
INPUT_NO_TEMPO=false
INPUT_NO_TIME_SIGNATURE=false

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=120
MAX_STEPS=210000
VALIDATION_INTERVAL=1000
VALIDATION_STEPS=1000
GRAD_NORM_CLIP=1.0
LEARNING_RATE_PEAK=0.00025
LEARNING_RATE_WARMUP_STEPS=5000
LEARNING_RATE_DECAY_END_STEPS=210000
LEARNING_RATE_DECAY_END_RATIO=0.0
MAX_PIECE_PER_GPU=3
EARLY_STOP=10

# others
USE_DEVICE='cuda'
# use accelerate by Huggingface
USE_PARALLEL=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
