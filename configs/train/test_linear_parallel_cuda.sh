#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=64
VIRTUAL_PIECE_STEP_RATIO=0.5
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
PITCH_AUGMENTATION_RANGE=0

# model parameter
USE_LINEAR_ATTENTION=true
LAYERS_NUMBER=1
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=32
NOT_USE_MPS_NUMBER=false

# training parameter
SPLIT_RATIO="-1 100"
BATCH_SIZE=64
MAX_UPDATES=200
VALIDATION_INTERVAL=10
LOSS_NONPAD_DIM=all
MAX_GRAD_NORM=0.0
LEARNING_RATE_PEAK=0.001
LEARNING_RATE_WARMUP_UPDATES=40
LEARNING_RATE_DECAY_END_UPDATES=160
LEARNING_RATE_DECAY_END_RATIO=0.1
EARLY_STOP=10

# device
USE_DEVICE='cuda'
USE_PARALLEL=true
MAX_PIECE_PER_GPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# eval
NO_EVAL=true
EVAL_MIDI_TO_PIECE_PARAS_FILE=""
EVAL_SAMPLE_NUMBER=10 # if not set, will used the number of test files
PRIMER_LENGTH=4
NUCLEUS_THRESHOLD=1.0
SOFTMAX_TEMPERATURE=1.0
