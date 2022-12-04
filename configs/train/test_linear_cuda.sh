#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=64
USE_PERMUTABLE_SUBSEQ_LOSS=false
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
PITCH_AUGMENTATION=0
SAMPLE_FROM_START=true

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=1
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=32
INPUT_NO_TEMPO=false
INPUT_NO_TIME_SIGNATURE=false

# training parameter
SPLIT_RATIO="-1 100"
BATCH_SIZE=8
STEPS=200
VALIDATION_INTERVAL=10
VALIDATION_STEPS=10
LOG_HEAD_LOSSES=true
GRAD_NORM_CLIP=1.0
LEARNING_RATE=0.001
LEARNING_RATE_WARMUP_STEPS=40
LEARNING_RATE_DECAY_END_STEPS=160
LEARNING_RATE_DECAY_END_RATIO=0.1
EARLY_STOP=10

# others
USE_DEVICE='cuda'
