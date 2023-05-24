#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=1024
MEASURE_SAMPLE_STEP_RATIO=1.0
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=false

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=12
ATTN_HEADS_NUMBER=16
EMBEDDING_DIM=512
INPUT_CONTEXT=false
INPUT_INSTRUMENTS=false
OUTPUT_INSTRUMENTS=false

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=128
MAX_UPDATES=210000
VALIDATION_INTERVAL=1000
LOSS_NONPAD_DIM=all
MAX_GRAD_NORM=0.0
LEARNING_RATE_PEAK=0.0005
LEARNING_RATE_WARMUP_UPDATES=5000
LEARNING_RATE_DECAY_END_UPDATES=210000
LEARNING_RATE_DECAY_END_RATIO=0.0
EARLY_STOP=10

# device
USE_DEVICE='cuda'
USE_PARALLEL=true
MAX_PIECE_PER_GPU=16
export CUDA_VISIBLE_DEVICES=0,1,2,3

# eval
EVAL_SAMPLE_NUMBER=100
PRIMER_LENGTH=4
NUCLEUS_THRESHOLD=0.95
SOFTMAX_TEMPERATURE="0.9 1.0 1.2 1.2 1.0 1.0"
