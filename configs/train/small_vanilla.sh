#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=1024
USE_PERMUTABLE_SUBSEQ_LOSS=false
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
SAMPLE_FROM_START=false

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=6
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
INPUT_NO_TEMPO=false
INPUT_NO_TIME_SIGNATURE=false

# training parameter
SPLIT_RATIO="-1 1000"
BATCH_SIZE=8
STEPS=1000000
VALIDATION_INTERVAL=10000
LOG_HEAD_LOSSES=false
GRAD_NORM_CLIP=1.0
LEARNING_RATE=0.001
LEARNING_RATE_WARMUP_STEPS=20000
LEARNING_RATE_DECAY_END_STEPS=80000
LEARNING_RATE_DECAY_END_RATIO=0.1
EARLY_STOP=10

# others
USE_DEVICE='cuda'
