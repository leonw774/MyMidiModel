#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=1024
MEASURE_SAMPLE_STEP_RATIO=0.5
PERMUTE_MPS=true
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=true

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=6
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
INPUT_CONTEXT=false
INPUT_INSTRUMENTS=false
OUTPUT_INSTRUMENTS=true

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=128
MAX_UPDATES=50000
VALIDATION_INTERVAL=1000
GENERATE_INTERVAL=10000
WEIGHT_LOSS_BY_NONPAD_NUM=true
GRAD_CLIP_NORM=0.0
LEARNING_RATE_PEAK=0.00001
LEARNING_RATE_WARMUP_UPDATES=2500
LEARNING_RATE_DECAY_END_UPDATES=50000
LEARNING_RATE_DECAY_END_RATIO=0.0
EARLY_STOP=10

# device
USE_DEVICE='cuda'
USE_PARALLEL=true
MAX_PIECE_PER_GPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# eval
EVAL_SAMPLE_NUMBER=100
PRIMER_LENGTH=4
NUCLEUS_THRESHOLD=1.0
