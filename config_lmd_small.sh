#!/bin/bash
# midi preprocessing
NTH=96
MAX_TRACK_NUMBER=32
MAX_DURATION=96
VELOCITY_STEP=8
TEMPO_MIN=24
TEMPO_MAX=200
TEMPO_STEP=16
POSITION_METHOD="event"
PROCESS_WORKERS=8
MIDI_DIR_PATH="data/midis/lmd_full/0"
MIDI_OTHER_ARGUMENTS="--make-stats"
PROC_DATA_NAME=lmd_small_nth${NTH}_r${MAX_TRACK_NUMBER}_d${MAX_DURATION}_v${VELOCITY_STEP}_t${TEMPO_MIN}_${TEMPO_MAX}_${TEMPO_STEP}

# text corpus and vocabulary
BPE_ITER=100
MAX_SAMPLE_LENGTH=1024
DATA_OTHER_ARGUMENTS='--debug'
