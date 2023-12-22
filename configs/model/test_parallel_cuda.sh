#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=64
VIRTUAL_PIECE_STEP_RATIO=1
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
PITCH_AUGMENTATION_RANGE=0

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=1
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=32
NOT_USE_MPS_NUMBER=false

# training parameter
BATCH_SIZE=8
MAX_UPDATES=200
VALIDATION_INTERVAL=100
LOSS_PADDING="ignore"
MAX_GRAD_NORM=1.0
LEARNING_RATE_PEAK=0.0001
LEARNING_RATE_WARMUP_UPDATES=100
LEARNING_RATE_DECAY_END_UPDATES=200
LEARNING_RATE_DECAY_END_RATIO=0.5
EARLY_STOP=0

# do eval with valid set?
VALID_EVAL_SAMPLE_NUMBER=0

# device
USE_DEVICE="cuda"
USE_PARALLEL=true
MAX_PIECE_PER_GPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# eval
EVAL_MIDI_TO_PIECE_PARAS_FILE=""
EVAL_SAMPLE_NUMBER=10 # if not set, will used the number of test files
EVAL_WORKER_NUMBER=32
ONLY_EVAL_UNCOND=true
PRIMER_LENGTH=4
SAMPLE_FUNCTION="none"
SAMPLE_THRESHOLD=1.0
SAMPLE_THRESHOLD_HEAD_MULTIPLIER=1.0
SOFTMAX_TEMPERATURE=1.0
