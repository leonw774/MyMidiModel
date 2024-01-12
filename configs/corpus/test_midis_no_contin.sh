#!/bin/bash
# midi preprocessing
TPQ=12
MAX_TRACK_NUMBER=40
MAX_DURATION=48
VELOCITY_STEP=4
CONTINUING_NOTE=false
TEMPO_MIN=8
TEMPO_MAX=240
TEMPO_STEP=8
MIDI_WORKER_NUMBER=1
MIDI_DIR_PATH="data/test_midis"
DATA_NAME="test_midis"
TEST_PATHS_FILE='configs/split/empty_test.txt'
VALID_PATHS_FILE='configs/split/empty_valid.txt'
MIDI_TO_CORPUS_VERBOSE=true
