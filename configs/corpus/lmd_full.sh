#!/bin/bash
# midi preprocessing
NTH=32
MAX_TRACK_NUMBER=40
MAX_DURATION=32
VELOCITY_STEP=16
CONTINUING_NOTE=true
USE_MERGE_DRUMS=true
TEMPO_MIN=24
TEMPO_MAX=200
TEMPO_STEP=16
WORKER_NUMBER=32
MIDI_DIR_PATH="data/midis/lmd_full"
DATA_NAME="lmd_full"
TEST_PATHS_FILE='configs/split/lmd_full_test.txt'
VALID_PATHS_FILE='configs/split/lmd_full_valid.txt'
