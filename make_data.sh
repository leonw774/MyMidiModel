#!/bin/bash
# check if first argument is a file
if [ -f "$1" ]; then
    if source $1; then
        echo "source $1: success"
    else
        echo "source $1: fail"
        exit 0
    fi
else
    echo "'$1' file not exists"
    exit 0
fi

echo $PROC_DATA_NAME

python3 midi_to_text.py --nth ${NTH} --max-track-number ${MAX_TRACK_NUMBER} --max-duration ${MAX_DURATION} --velocity-step ${VELOCITY_STEP} \
    --tempo-quantization ${TEMPO_MIN} ${TEMPO_MAX} ${TEMPO_STEP} --tempo-method ${TEMPO_METHOD} ${MIDI_OTHER_ARGUMENTS} \
    -w ${PROCESS_WORKERS} -r -o data/corpus/${PROC_DATA_NAME}.txt ${MIDI_DIR_PATH}

test $? -ne 0 || python3 make_data.py ${DATA_OTHER_ARGUMENTS} --max-sample-length ${MAX_SAMPLE_LENGTH} \
    data/corpus/${PROC_DATA_NAME}.txt  data/data/${PROC_DATA_NAME}
