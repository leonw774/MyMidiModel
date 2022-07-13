#!/bin/bash
# midi preprocessing
NTH=96
MAX_TRACK_NUMBER=24
MAX_DURATION=4
VELOCITY_STEP=16
TEMPO_MIN=24
TEMPO_MAX=200
TEMPO_STEP=16
TEMPO_METHOD="position_event"
PROCESS_WORKERS=1
MIDI_DIR_PATH="data/test_midis"
MIDI_OTHER_ARGUMENTS="--verbose"
PROC_DATA_NAME="test_midis"

# make vocabulary
MAX_SAMPLE_LENGTH=1024
DATA_OTHER_ARGUMENTS='--debug'