#!/bin/bash
SEED=413

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
INPUT_CONTEXT=true
INPUT_INSTRUMENTS=true
OUTPUT_INSTRUMENTS=true

# training parameter
SPLIT_RATIO="-1 100"
BATCH_SIZE=8
MAX_UPDATES=200
VALIDATION_INTERVAL=10
LOSS_NONPAD_DIM=none
MAX_GRAD_NORM=0.0
LEARNING_RATE_PEAK=0.0001
LEARNING_RATE_WARMUP_UPDATES=40
LEARNING_RATE_DECAY_END_UPDATES=160
LEARNING_RATE_DECAY_END_RATIO=0.1
EARLY_STOP=10

# device
USE_DEVICE='cpu'

# eval
NO_EVAL=true
EVAL_SAMPLE_NUMBER=10
PRIMER_LENGTH=4
NUCLEUS_THRESHOLD=1.0
SOFTMAX_TEMPERATURE=1.0
