#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=1024
MEASURE_SAMPLE_STEP_RATIO=0.25
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=false
WEIGHT_LOSS_BY_NONPAD_NUM=true

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=6
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=256
INPUT_NO_TEMPO=false
INPUT_NO_TIME_SIGNATURE=false

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=128
MAX_STEPS=210000
VALIDATION_INTERVAL=1000
VALIDATION_STEPS=1000
GRAD_NORM_CLIP=1.0
LEARNING_RATE_PEAK=0.0003
LEARNING_RATE_WARMUP_STEPS=5000
LEARNING_RATE_DECAY_END_STEPS=210000
LEARNING_RATE_DECAY_END_RATIO=0.0
MAX_PIECE_PER_GPU=8
EARLY_STOP=10

# device
USE_DEVICE='cuda'
# use accelerate by Huggingface
USE_PARALLEL=true
export CUDA_VISIBLE_DEVICES=0,1

# eval
EVAL_SAMPLE_NUMBER=100
NUCLEUS_THRESHOLD=1.0
