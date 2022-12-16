#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=1024
USE_PERMUTABLE_SUBSEQ_LOSS=false
PERMUTE_MPS=true
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION=6
SAMPLE_FROM_START=true

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=12
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
INPUT_NO_TEMPO=false
INPUT_NO_TIME_SIGNATURE=false

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=64
STEPS=400000
VALIDATION_INTERVAL=5000
VALIDATION_STEPS=10000
LOG_HEAD_LOSSES=false
GRAD_NORM_CLIP=1.0
LEARNING_RATE=0.0001
LEARNING_RATE_WARMUP_STEPS=20000
LEARNING_RATE_DECAY_END_STEPS=400000
LEARNING_RATE_DECAY_END_RATIO=0.0
EARLY_STOP=10

# others
USE_DEVICE='cuda'
# use accelerate by Huggingface
USE_PARALLEL=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
