#!/bin/bash
SEED=413

# generation setting
EVAL_SAMPLE_NUMBER="" # if not set, will used the number of test files
EVAL_WORKER_NUMBER=32
PRIMER_MEASURE_LENGTH=4
SAMPLE_FUNCTION="top-p"
SAMPLE_THRESHOLD=0.95
SOFTMAX_TEMPERATURE=1.0

# evaluation setting
EVAL_MIDI_TO_PIECE_PARAS_FILE=""