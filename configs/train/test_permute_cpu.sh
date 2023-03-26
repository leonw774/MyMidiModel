#!/bin/bash
# dataset parameters
MAX_SEQ_LENGTH=64
MEASURE_SAMPLE_STEP_RATIO=0.5
PERMUTE_MPS=true
PERMUTE_TRACK_NUMBER=true
PITCH_AUGMENTATION_RANGE=0
USE_PERMUTABLE_SUBSEQ_LOSS=true

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=1
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=32
INPUT_INSTRUMENTS=false
OUTPUT_INSTRUMENTS=true

# training parameter
SPLIT_RATIO="-1 100"
BATCH_SIZE=8
MAX_STEPS=200
VALIDATION_INTERVAL=10
VALIDATION_STEPS=10
GEN_SAMPLE_INTERVAL=50
WEIGHT_LOSS_BY_NONPAD_NUM=true
GRAD_CLIP_NORM=0.0
LEARNING_RATE_PEAK=0.001
LEARNING_RATE_WARMUP_STEPS=40
LEARNING_RATE_DECAY_END_STEPS=160
LEARNING_RATE_DECAY_END_RATIO=0.1
EARLY_STOP=10

# device
USE_DEVICE='cpu'

# eval
EVAL_SAMPLE_NUMBER=10
PRIMER_LENGTH=4
NUCLEUS_THRESHOLD=1.0
