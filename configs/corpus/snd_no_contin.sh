#!/bin/bash
# midi preprocessing
TPQ=12
MAX_TRACK_NUMBER=40
MAX_DURATION=48
VELOCITY_STEP=4
CONTINUING_NOTE=false
USE_MERGE_DRUMS=false
TEMPO_MIN=8
TEMPO_MAX=240
TEMPO_STEP=8
WORKER_NUMBER=32
MIDI_DIR_PATH="data/midis/SymphonyNet_Dataset"
DATA_NAME="snd"
TEST_PATHS_FILE='configs/split/snd_test.txt'
VALID_PATHS_FILE='configs/split/snd_valid.txt'
