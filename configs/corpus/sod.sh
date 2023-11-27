#!/bin/bash
# midi preprocessing
TPQ=12
MAX_TRACK_NUMBER=40
MAX_DURATION=48
VELOCITY_STEP=4
CONTINUING_NOTE=true
USE_MERGE_DRUMS=false
TEMPO_MIN=8
TEMPO_MAX=240
TEMPO_STEP=8
WORKER_NUMBER=8
MIDI_DIR_PATH="data/midis/Symbolic_Orchestral_Dataset"
DATA_NAME="sod"
TEST_PATHS_FILE='configs/split/sod_test.txt'
VALID_PATHS_FILE='configs/split/sod_valid.txt'
