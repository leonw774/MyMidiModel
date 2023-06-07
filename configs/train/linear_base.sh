#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=1024
VIRTUAL_PIECE_STEP_RATIO=0.25
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=false

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=12
ATTN_HEADS_NUMBER=16
EMBEDDING_DIM=512
NOT_USE_MPS_NUMBER=false
INPUT_CONTEXT=false
INPUT_INSTRUMENTS=false
OUTPUT_INSTRUMENTS=false

# training parameter
SPLIT_RATIO="95 5"
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
NUCLEUS_THRESHOLD=0.99
SOFTMAX_TEMPERATURE=1.0
