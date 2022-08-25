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
LOG_PATH="./logs/$(date '+%Y%m%d%H%M')_$PROC_DATA_NAME.log"
echo "log file: $LOG_PATH"
touch $LOG_PATH

python3 midi_to_text.py --nth $NTH --max-track-number $MAX_TRACK_NUMBER --max-duration $MAX_DURATION --velocity-step $VELOCITY_STEP \
    --tempo-quantization $TEMPO_MIN $TEMPO_MAX $TEMPO_STEP --position-method $POSITION_METHOD $MIDI_OTHER_ARGUMENTS \
    --log $LOG_PATH -w $PROCESS_WORKERS -r -o data/corpus/$PROC_DATA_NAME $MIDI_DIR_PATH
test $? -ne 0 && echo "midi_to_text.py failed. make_data.sh exit." && exit 1

if [ "$BPE_ITER" -ne "0" ]; then
    echo "start learn bpe vocab"

    # compile
    g++ bpe/learn_vocab.cpp -o bpe/learn_vocab -fopenmp
    test $? -ne 0 && echo "bpe/learn_vocab.cpp compile error. make_data.sh exit." && exit 1

    mkdir data/corpus/${PROC_DATA_NAME}_bpeiter${BPE_ITER}
    cp data/corpus/${PROC_DATA_NAME}/paras data/corpus/${PROC_DATA_NAME}_bpeiter${BPE_ITER}/paras
    # use tee command to copy stdout to log
    bpe/learn_vocab $BPE_ITER data/corpus/${PROC_DATA_NAME} | tee -a $LOG_PATH
    test $PIPESTATUS[0] -ne 0 && echo "learn_vocab failed. make_data.sh exit." && exit 1

    # check if tokenized corpus is equal to original corpus
    # sample size of 100 seem to be enough
    python3 verify_corpus_equality.py data/corpus/${PROC_DATA_NAME} data/corpus/${PROC_DATA_NAME}_bpeiter${BPE_ITER} 100
    test $? -ne 0 && echo "make_data.sh exit." && exit 1
fi

python3 text_to_array.py $DATA_OTHER_ARGUMENTS --max-sample-length $MAX_SAMPLE_LENGTH --bpe $BPE_ITER --log $LOG_PATH \
    data/corpus/$PROC_DATA_NAME
