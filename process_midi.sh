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

echo $PROCFILE_SUFFIX

python3 data/midi_2_text.py --nth ${NTH} --max-track-number ${MAX_TRACK_NUMBER} --max-duration ${MAX_DURATION} \
    --velocity-step ${VELOCITY_STEP} --tempo-quantization ${TEMPO_MIN} ${TEMPO_MAX} ${TEMPO_STEP} ${ADDITIONAL_ARGUMENTS} \
    -w ${PROCESS_WORKERS} -r -o data/processed_midi/corpus_${PROCFILE_SUFFIX}.txt ${MIDI_DIR_PATH}

python3 data/make_data.py --max-sample-length ${MAX_SAMPLE_LENGTH} \
    data/processed_midi/corpus_${PROCFILE_SUFFIX}.txt \
    data/data/${PROCFILE_SUFFIX}
