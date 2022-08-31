#!/bin/bash
echo "make_data.sh start."
if [ $# -ne 3 ]; then
    echo "Expect arguments to be three configuration file name for midi, bpe and model."
fi

# check if all argument is a file and execute them to get their vars
FULL_CONFIG_NAME=$1"-"$2"-"$3
MIDI_CONFIG="configs/midi/"$1".sh"
BPE_CONFIG="configs/bpe/"$2".sh"
MODEL_CONFIG="configs/model/"$3".sh"

for CONFIG_PATH in $MIDI_CONFIG $BPE_CONFIG $MODEL_CONFIG
do
    if [ -f "$CONFIG_PATH" ]; then
        if source $CONFIG_PATH; then
            echo "source $CONFIG_PATH: success"
        else
            echo "source $CONFIG_PATH: fail"
            exit 1
        fi
    else
        echo "'$CONFIG_PATH' file not exists"
        exit 1
    fi
done

LOG_PATH="./logs/$(date '+%Y%m%d%H%M')-${FULL_CONFIG_NAME}.log"
echo "Log file: $LOG_PATH"
touch $LOG_PATH

CORPUS_DIR_PATH="data/corpus/${DATA_NAME}_nth${NTH}_r${MAX_TRACK_NUMBER}_d${MAX_DURATION}_v${VELOCITY_STEP}_t${TEMPO_MIN}_${TEMPO_MAX}_${TEMPO_STEP}_pos${POSITION_METHOD}"
if [ $USE_EXISTED == true ]; then
    MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-existed"
    echo "Appended --use-existed"
fi
if [ $CONTINUING_NOTE == true ]; then
    MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-continuing-note"
    echo "Appended --use-continuing-note"
fi

python3 midi_to_text.py --nth $NTH --max-track-number $MAX_TRACK_NUMBER --max-duration $MAX_DURATION --velocity-step $VELOCITY_STEP \
    --tempo-quantization $TEMPO_MIN $TEMPO_MAX $TEMPO_STEP --position-method $POSITION_METHOD $MIDI_TO_TEXT_OTHER_ARGUMENTS \
    --log $LOG_PATH -w $PROCESS_WORKERS -r -o $CORPUS_DIR_PATH $MIDI_DIR_PATH
test $? -ne 0 && { echo "midi_to_text.py failed. make_data.sh exit." | tee -a $LOG_PATH ; } && exit 1


if [ $BPE_ITER -ne 0 ]; then
    echo "start learn bpe vocab"
    CORPUS_DIR_PATH_WITH_BPE="${CORPUS_DIR_PATH}_bpe${BPE_ITER}_${SHAPECOUNT_METHOD}_${SHAPECOUNT_SAMPLERATE}"

    # compile
    make -C ./bpe
    test $? -ne 0 && { echo "learn_vocab compile error. make_data.sh exit." | tee -a $LOG_PATH ; } && exit 1

    # create new dir 
    if [ -d $CORPUS_DIR_PATH_WITH_BPE ]; then
        rm -f ${CORPUS_DIR_PATH_WITH_BPE}/*
    else
        mkdir $CORPUS_DIR_PATH_WITH_BPE
    fi

    # copy paras and pathlist
    cp "./${CORPUS_DIR_PATH}/paras" $CORPUS_DIR_PATH_WITH_BPE
    cp "./${CORPUS_DIR_PATH}/pathlist" $CORPUS_DIR_PATH_WITH_BPE

    # run learn_vocab and use tee command to copy stdout to log
    bpe/learn_vocab $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE $BPE_ITER $SHAPECOUNT_METHOD $SHAPECOUNT_SAMPLERATE | tee -a $LOG_PATH
    BPE_EXIT_CODE=${PIPESTATUS[0]}
    test $BPE_EXIT_CODE -ne 0 && { echo "learn_vocab failed. exit code: $BPE_EXIT_CODE. make_data.sh exit." | tee -a $LOG_PATH ; } && exit 1

    # check if tokenized corpus is equal to original corpus
    # this is buggy so not using it
    # python3 verify_corpus_equality.py $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE 100 | tee -a $LOG_PATH
    # test $? -ne 0 && echo "make_data.sh exit." && exit 1

    # replace CORPUS_DIR_PATH to CORPUS_DIR_PATH_WITH_BPE
    CORPUS_DIR_PATH=$CORPUS_DIR_PATH_WITH_BPE
fi

python3 text_to_array.py --max-seq-length $MAX_SEQ_LENGTH --bpe $BPE_ITER --log $LOG_PATH $TEXT_TO_ARRAY_OTHER_ARGUMENTS $CORPUS_DIR_PATH

echo "make_data.sh start."
