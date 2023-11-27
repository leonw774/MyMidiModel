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
WORKER_NUMBER=32
MIDI_DIR_PATH="data/midis/lmd_full"
DATA_NAME="lmd_full"
TEST_PATHS_FILE='configs/split/lmd_full_test.txt'
VALID_PATHS_FILE='configs/split/lmd_full_valid.txt'
