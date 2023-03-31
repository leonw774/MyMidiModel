#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=1024
MEASURE_SAMPLE_STEP_RATIO=0.5
PERMUTE_MPS=true
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=false

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=6
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=512
INPUT_CONTEXT=false
INPUT_INSTRUMENTS=false
OUTPUT_INSTRUMENTS=false

# training parameter
SPLIT_RATIO="99 1"
BATCH_SIZE=128
MAX_STEPS=210000
VALIDATION_INTERVAL=1000
VALIDATION_STEPS=1000
GEN_SAMPLE_INTERVAL=10000
WEIGHT_LOSS_BY_NONPAD_NUM=true
GRAD_CLIP_NORM=0.0
LEARNING_RATE_PEAK=0.001
LEARNING_RATE_WARMUP_STEPS=5000
LEARNING_RATE_DECAY_END_STEPS=100000
LEARNING_RATE_DECAY_END_RATIO=0.1
MAX_PIECE_PER_GPU=16
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
