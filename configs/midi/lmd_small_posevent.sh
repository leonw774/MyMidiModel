#!/bin/bash
# midi preprocessing
NTH=96
MAX_TRACK_NUMBER=32
MAX_DURATION=96
VELOCITY_STEP=16
CONTINUING_NOTE=true
TEMPO_MIN=24
TEMPO_MAX=200
TEMPO_STEP=16
POSITION_METHOD="event"
PROCESS_WORKERS=16
MIDI_DIR_PATH="data/midis/lmd_full/0"
DATA_NAME="lmd_small"