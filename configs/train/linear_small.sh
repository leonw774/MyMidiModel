#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=2048
MEASURE_SAMPLE_STEP_RATIO=0.25
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=false

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
WEIGHT_LOSS_BY_NONPAD_NUM=true
GRAD_CLIP_NORM=0.0
LEARNING_RATE_PEAK=0.0001
LEARNING_RATE_WARMUP_STEPS=5000
LEARNING_RATE_DECAY_END_STEPS=210000
LEARNING_RATE_DECAY_END_RATIO=0.0
MAX_PIECE_PER_GPU=11
EARLY_STOP=10

# device
USE_DEVICE='cuda'
# use accelerate by Huggingface
USE_PARALLEL=true
export CUDA_VISIBLE_DEVICES=0,1,2,3

# eval
EVAL_SAMPLE_NUMBER=100
PRIMER_LENGTH=4
NUCLEUS_THRESHOLD=1.0
