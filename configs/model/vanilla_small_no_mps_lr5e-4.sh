#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=1024
VIRTUAL_PIECE_STEP_RATIO=0.5
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION_RANGE=0

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=6
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
NOT_USE_MPS_NUMBER=true

# training parameter
BATCH_SIZE=128
MAX_UPDATES=200000
VALIDATION_INTERVAL=1000
LOSS_PADDING="ignore"
MAX_GRAD_NORM=1.0
LEARNING_RATE_PEAK=0.0005
LEARNING_RATE_WARMUP_UPDATES=5000
LEARNING_RATE_DECAY_END_UPDATES=200000
LEARNING_RATE_DECAY_END_RATIO=0.0
EARLY_STOP=10

# do eval with valid set?
VALID_EVAL_SAMPLE_NUMBER=0

# device
USE_DEVICE="cuda"
USE_PARALLEL=true
MAX_PIECE_PER_GPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# eval
EVAL_MIDI_TO_PIECE_PARAS_FILE=""
EVAL_SAMPLE_NUMBER="" # if not set, will used the number of test files
PRIMER_LENGTH=4
SAMPLE_FUNCTION="top-p"
SAMPLE_THRESHOLD=0.95
SOFTMAX_TEMPERATURE=1.0
