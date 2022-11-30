#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=4096
USE_PERMUTABLE_SUBSEQ_LOSS=false
PERMUTE_MPS=true
PERMUTE_TRACK_NUMBER=true
SAMPLE_FROM_START=false

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=12
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
INPUT_NO_TEMPO=false
INPUT_NO_TIME_SIGNATURE=false

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=64
STEPS=200000
VALIDATION_INTERVAL=5000
VALIDATION_STEPS=5000
LOG_HEAD_LOSSES=false
GRAD_NORM_CLIP=1.0
LEARNING_RATE=0.0002
LEARNING_RATE_WARMUP_STEPS=5000
LEARNING_RATE_DECAY_END_STEPS=200000
LEARNING_RATE_DECAY_END_RATIO=0.0
EARLY_STOP=10

# others
USE_DEVICE='cuda'
