#!/bin/bash
echo "pipeline.sh start."
USE_EXISTED=""
if [ $# -eq 4 ]; then
    if [ $4 == '--use-existed' ]; then
        USE_EXISTED="--use-existed"
    else
        echo "Expect arguments to be three configuration file name for midi preprocessing, bpe setting and training/model setting, and an optional '--use-existed' flag at the fourth position."
        exit 1
    fi
else
    if [ $# -ne 3 ]; then
        echo "Expect arguments to be three configuration file name for midi preprocessing, bpe setting and training/model setting, and an optional '--use-existed' flag at the fourth position."
        exit 1
    fi
fi

# check if all argument is a file and execute them to get their vars
FULL_CONFIG_NAME=$1"-"$2"-"$3
MIDI_CONFIG="configs/midi/"$1".sh"
BPE_CONFIG="configs/bpe/"$2".sh"
TRAIN_CONFIG="configs/train/"$3".sh"

for CONFIG_PATH in $MIDI_CONFIG $BPE_CONFIG $TRAIN_CONFIG
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

LOG_PATH="logs/$(date '+%Y%m%d%H%M')-${FULL_CONFIG_NAME}.log"
echo "Corpus log file: $LOG_PATH"
touch $LOG_PATH

CORPUS_DIR_PATH="data/corpus/${DATA_NAME}_nth${NTH}_r${MAX_TRACK_NUMBER}_d${MAX_DURATION}_v${VELOCITY_STEP}_t${TEMPO_MIN}_${TEMPO_MAX}_${TEMPO_STEP}_pos${POSITION_METHOD}"

if [ $CONTINUING_NOTE == true ]; then
    MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-continuing-note"
    echo "Appended --use-continuing-note to midi_to_text's argument" | tee -a $LOG_PATH
fi

python3 midi_to_text.py --nth $NTH --max-track-number $MAX_TRACK_NUMBER --max-duration $MAX_DURATION --velocity-step $VELOCITY_STEP \
    --tempo-quantization $TEMPO_MIN $TEMPO_MAX $TEMPO_STEP --position-method $POSITION_METHOD $MIDI_TO_TEXT_OTHER_ARGUMENTS $USE_EXISTED \
    --log $LOG_PATH -w $PROCESS_WORKERS -r -o $CORPUS_DIR_PATH $MIDI_DIR_PATH
test $? -ne 0 && { echo "midi_to_text.py failed. make_data.sh exit." | tee -a $LOG_PATH ; } && exit 1


if [ $BPE_ITER -ne 0 ]; then
    echo "Start learn bpe vocab" | tee -a $LOG_PATH
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

    # this is buggy so not using it
    # check if tokenized corpus is equal to original corpus
    # python3 verify_corpus_equality.py $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE 100 | tee -a $LOG_PATH
    # test $? -ne 0 && echo "make_data.sh exit." && exit 1

    # replace CORPUS_DIR_PATH to CORPUS_DIR_PATH_WITH_BPE
    CORPUS_DIR_PATH=$CORPUS_DIR_PATH_WITH_BPE
fi

python3 text_to_array.py --bpe $BPE_ITER --log $LOG_PATH --debug $CORPUS_DIR_PATH $USE_EXISTED

# test if NO_TRAIN is a set variables
if [ -n "${NO_TRAIN+x}" ]; then
    echo "Not training" | tee -a $LOG_PATH
    echo "pipeline.sh end"
    exit 0
fi

CHECKPOINT_DIR_PATH="ckpt/$(date '+%Y%m%d%H%M')-"$FULL_CONFIG_NAME
if [ -d $CHECKPOINT_DIR_PATH ]; then
        rm -rf ${CHECKPOINT_DIR_PATH}
    else
        mkdir $CHECKPOINT_DIR_PATH
    fi
echo "Checkpoint dir: $CHECKPOINT_DIR_PATH"

TRAIN_OTHER_ARGUMENTS=""
if [ $USE_PERMUTABL_SUBSEQ_LOSS == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-permutable-subseq-loss"
    echo "Appended --use-permutable-subseq-loss to train's argument" | tee -a $LOG_PATH
fi
if [ $PERMUTE_MPS == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-mps"
    echo "Appended --permute-mps to train's argument" | tee -a $LOG_PATH
fi
if [ $PERMUTE_TRACK_NUMBER == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-track-number"
    echo "Appended --permute-track-number to train's argument" | tee -a $LOG_PATH
fi
if [ $LOG_HEAD_LOSSES == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --log-head-losses"
    echo "Appended --log-head-losses to train's argument" | tee -a $LOG_PATH
fi


python3 train.py --max-seq-length $MAX_SEQ_LENGTH --sample-stride $SAMPLE_STRIDE \
    --layers-number $LAYERS_NUMBER --attn-heads-number $ATTN_HEADS_NUMBER --embedding-dim $EMBEDDING_DIM \
    --split-ratio $SPLIT_RATIO --batch-size $BATCH_SIZE --steps $STEPS --validation-interval $VALIDATION_INTERVAL --grad-norm-clip $GRAD_NORM_CLIP --early-stop-tolerance $EARLY_STOP_TOLERANCE $TRAIN_OTHER_ARGUMENTS \
    --lr $LEARNING_RATE --lr-warmup-steps $LEARNING_RATE_WARMUP_STEPS --lr-decay-end-steps $LEARNING_RATE_DECAY_END_STEPS --lr-decay-end-ratio $LEARNING_RATE_DECAY_END_RATIO \
    --use-device $USE_DEVICE --log $LOG_PATH --checkpoint-dir-path $CHECKPOINT_DIR_PATH $CORPUS_DIR_PATH

echo "pipeline.sh end"
