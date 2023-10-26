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
WORKER_NUMBER=1
MIDI_DIR_PATH="data/test_midis/multi-note_bpe_example"
DATA_NAME="mnbpe_example"
TEST_PATHS_FILE='configs/split/empty_test.txt'
VALID_PATHS_FILE='configs/split/empty_valid.txt'
MIDI_TO_CORPUS_VERBOSE=true
