#!/bin/bash
SEED=413

# generation setting
EVAL_SAMPLE_NUMBER=10 # if not set, will used the number of test files
EVAL_WORKER_NUMBER=32
ONLY_EVAL_UNCOND=true
PRIMER_LENGTH=4
SAMPLE_FUNCTION="none"
SAMPLE_THRESHOLD=1.0
SOFTMAX_TEMPERATURE=1.0

# evaluation setting
EVAL_MIDI_TO_PIECE_PARAS_FILE=""
