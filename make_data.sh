#!/bin/bash
# check if first argument is a file
if [ -f "$1" ]; then
    if source $1; then
        echo "source $1: success"
    else
        echo "source $1: fail"
        exit 1
    fi
else
    echo "'$1' file not exists"
    exit 1
fi

echo $PROC_DATA_NAME
LOG_PATH="./logs/$(date '+%Y%m%d%H%M')_${PROC_DATA_NAME}.log"
echo "log file: $LOG_PATH"
touch $LOG_PATH

# python3 midi_to_text.py --nth ${NTH} --max-track-number ${MAX_TRACK_NUMBER} --max-duration ${MAX_DURATION} --velocity-step ${VELOCITY_STEP} \
#     --tempo-quantization ${TEMPO_MIN} ${TEMPO_MAX} ${TEMPO_STEP} --tempo-method ${TEMPO_METHOD} ${MIDI_OTHER_ARGUMENTS} \
#     --log ${LOG_PATH} -w ${PROCESS_WORKERS} -r -o data/corpus/${PROC_DATA_NAME}.txt ${MIDI_DIR_PATH}

if [ $? -ne 0 ]; then
    echo "make_data.sh exit"
    exit 1
fi

if [ "$BPE_ITER" -ne "0" ]; then
    echo "make bpe"

    # compile if not existed
    g++ ./bpe/bpe.cpp -o ./bpe/bpe -fopenmp

    # create *_bpe.txt file if not existed
    touch data/corpus/${PROC_DATA_NAME}_bpe.txt

    # to overwrite it with the head matter of original corpus
    # this command find the line number n that start with '-', and copy first n lines to *_bpe.txt
    sed -n '/^-/{=;q;}' data/corpus/${PROC_DATA_NAME}.txt | (xargs head data/corpus/${PROC_DATA_NAME}.txt -n) > data/corpus/${PROC_DATA_NAME}_bpe.txt

    # use tee command to copy stdout and stderr to log
    ./bpe/bpe ${BPE_ITER} data/corpus/${PROC_DATA_NAME}.txt | tee -a ${LOG_PATH}

    if [ $? -ne 0 ]; then
        echo "make_data.sh exit"
        exit 1
    fi
fi

test $? -eq 0 && python3 make_data.py ${DATA_OTHER_ARGUMENTS} --max-sample-length ${MAX_SAMPLE_LENGTH} --bpe ${BPE_ITER} --log ${LOG_PATH}\
    data/corpus/${PROC_DATA_NAME}.txt  data/data/${PROC_DATA_NAME}
