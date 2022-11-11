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
STEPS=1000000
VALIDATION_INTERVAL=10000
VALIDATION_STEPS=10000
LOG_HEAD_LOSSES=false
GRAD_NORM_CLIP=1.0
LEARNING_RATE=0.003
LEARNING_RATE_WARMUP_STEPS=10000
LEARNING_RATE_DECAY_END_STEPS=1000000
LEARNING_RATE_DECAY_END_RATIO=0.01
EARLY_STOP=10

# others
USE_DEVICE='cuda'
