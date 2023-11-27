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
WORKER_NUMBER=1
MIDI_DIR_PATH="data/test_midis/multi-note_bpe_example"
DATA_NAME="mnbpe_example"
TEST_PATHS_FILE='configs/split/empty_test.txt'
VALID_PATHS_FILE='configs/split/empty_valid.txt'
MIDI_TO_CORPUS_VERBOSE=true
