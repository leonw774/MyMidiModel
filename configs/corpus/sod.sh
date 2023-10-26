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
WORKER_NUMBER=8
MIDI_DIR_PATH="data/midis/Symbolic_Orchestral_Dataset"
DATA_NAME="sod"
TEST_PATHS_FILE='configs/split/sod_test.txt'
VALID_PATHS_FILE='configs/split/sod_valid.txt'
