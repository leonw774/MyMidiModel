#!/bin/bash
# midi preprocessing
NTH=32
MAX_TRACK_NUMBER=40
MAX_DURATION=32
VELOCITY_STEP=16
CONTINUING_NOTE=false
USE_MERGE_DRUMS=true
TEMPO_MIN=24
TEMPO_MAX=200
TEMPO_STEP=16
PROCESS_WORKERS=32
MIDI_DIR_PATH="data/midis/lmd_full"
DATA_NAME="lmd_full"
TEST_PATHLIST='configs/test/lmd_full_test_pathlist.txt'
