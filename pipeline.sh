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

LOG_PATH="logs/$(date '+%Y%m%d-%H%M%S')-${FULL_CONFIG_NAME}.log"
echo "Log file: $LOG_PATH"
touch $LOG_PATH

CORPUS_DIR_PATH="data/corpus/${DATA_NAME}_nth${NTH}_r${MAX_TRACK_NUMBER}_d${MAX_DURATION}_v${VELOCITY_STEP}_t${TEMPO_MIN}_${TEMPO_MAX}_${TEMPO_STEP}_pos${POSITION_METHOD}"

MIDI_TO_TEXT_OTHER_ARGUMENTS=""
test "$CONTINUING_NOTE" == true && MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-continuing-note"
test "$USE_MERGE_DRUMS" == true && MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-merge-drums"
test "$MIDI_TO_TEXT_VERBOSE" == true && MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --verbose"
test -n "$TRAIN_OTHER_ARGUMENTS" && { echo "Appended ${MIDI_TO_TEXT_OTHER_ARGUMENTS} to midi_to_text's argument" | tee -a $LOG_PATH ; }

echo "Corpus dir: ${CORPUS_DIR_PATH}"

python3 midi_to_text.py --nth $NTH --max-track-number $MAX_TRACK_NUMBER --max-duration $MAX_DURATION --velocity-step $VELOCITY_STEP \
    --tempo-quantization $TEMPO_MIN $TEMPO_MAX $TEMPO_STEP --position-method $POSITION_METHOD $MIDI_TO_TEXT_OTHER_ARGUMENTS $USE_EXISTED \
    --log $LOG_PATH -w $PROCESS_WORKERS -r -o $CORPUS_DIR_PATH $MIDI_DIR_PATH
test $? -ne 0 && { echo "midi_to_text.py failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1


if [ $BPE_ITER -ne 0 ]; then
    echo "Start learn bpe vocab" | tee -a $LOG_PATH
    CORPUS_DIR_PATH_WITH_BPE="${CORPUS_DIR_PATH}_bpe${BPE_ITER}_${SCORING}_${MERGE_CONDITION}_${SAMPLE_RATE}"
    
    if [ -n "${USE_EXISTED}" ] && [ -d $CORPUS_DIR_PATH_WITH_BPE ] && [ -f "${CORPUS_DIR_PATH_WITH_BPE}/corpus" ] && [ -f "${CORPUS_DIR_PATH_WITH_BPE}/shape_vocab" ]; then
        echo "Output directory: ${CORPUS_DIR_PATH_WITH_BPE} already has corpus and shape_vocab file." | tee -a $LOG_PATH
        echo "Flag --use-existed is set" | tee -a $LOG_PATH
        echo "Learn bpe vocab is skipped" | tee -a $LOG_PATH
    else
        # compile
        make -C ./bpe
        test $? -ne 0 && { echo "Compile error. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1

        # create new dir 
        if [ -d $CORPUS_DIR_PATH_WITH_BPE ]; then
            echo "Output directory: ${CORPUS_DIR_PATH_WITH_BPE} already existed. Removed."
            rm -f "${CORPUS_DIR_PATH_WITH_BPE}/*"
        else
            mkdir $CORPUS_DIR_PATH_WITH_BPE
        fi

        # copy paras and pathlist
        cp "${CORPUS_DIR_PATH}/paras" $CORPUS_DIR_PATH_WITH_BPE
        cp "${CORPUS_DIR_PATH}/pathlist" $CORPUS_DIR_PATH_WITH_BPE

        # run learn_vocab
        bpe/learn_vocab $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE $BPE_ITER $SCORING $MERGE_CONDITION $SAMPLE_RATE $MIN_SCORE_LIMIT $BPE_VERBOSE | tee -a $LOG_PATH
        
        BPE_EXIT_CODE=${PIPESTATUS[0]}
        if [ $BPE_EXIT_CODE -ne 0 ]; then
            echo "learn_vocab failed. exit code: $BPE_EXIT_CODE. pipeline.sh exit." | tee -a $LOG_PATH
            rm -r "${CORPUS_DIR_PATH_WITH_BPE}"
            exit 1
        fi

        # process bpe log
        echo "sed -i 's/\r/\n/g ; s/\x1B\[2K//g' ${LOG_PATH}"
        sed -i 's/\r/\n/g ; s/\x1B\[2K//g' $LOG_PATH
        python3 plot_bpe_log.py $CORPUS_DIR_PATH_WITH_BPE $LOG_PATH

        # check if tokenized corpus is equal to original corpus
        python3 verify_corpus_equality.py $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE | tee -a $LOG_PATH
        VERIFY_EXIT_CODE=${PIPESTATUS[0]}
        test $VERIFY_EXIT_CODE -ne 0 && echo "Corpus equality verification failed. pipeline.sh exit." && exit 1
    fi

    # replace CORPUS_DIR_PATH to CORPUS_DIR_PATH_WITH_BPE
    CORPUS_DIR_PATH=$CORPUS_DIR_PATH_WITH_BPE
fi

python3 text_to_array.py --bpe $BPE_ITER --log $LOG_PATH --debug $CORPUS_DIR_PATH $USE_EXISTED
test $? -ne 0 && { echo "text_to_array.py failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1

# test if NO_TRAIN is a set variables
if [ -n "${NO_TRAIN+x}" ]; then
    echo "Not training" | tee -a $LOG_PATH
    echo "pipeline.sh exit."
    exit 0
fi

MODEL_DIR_PATH="models/$(date '+%Y%m%d-%H%M%S')-"$FULL_CONFIG_NAME
if [ -d $MODEL_DIR_PATH ]; then
        rm -rf ${MODEL_DIR_PATH}
    else
        mkdir $MODEL_DIR_PATH
        mkdir "${MODEL_DIR_PATH}/ckpt"
        mkdir "${MODEL_DIR_PATH}/eval_samples"
    fi
echo "Model dir: $MODEL_DIR_PATH"

TRAIN_OTHER_ARGUMENTS=""
test "$USE_PERMUTABLE_SUBSEQ_LOSS" == true && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-permutable-subseq-loss"
test "$PERMUTE_MPS" == true                && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-mps"
test "$PERMUTE_TRACK_NUMBER" == true       && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-track-number"
test "$SAMPLE_FROM_START" == true          && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --sample-from-start"
test "$USE_LINEAR_ATTENTION" == true       && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-linear-attn"
test "$INPUT_NO_TEMPO" == true             && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --input-no-tempo"
test "$INPUT_NO_TIME_SIGNATURE" == true    && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --input-no-time-signatrue"
test "$LOG_HEAD_LOSSES" == true            && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --log-head-losses"
test -n "$TRAIN_OTHER_ARGUMENTS" && { echo "Appended${TRAIN_OTHER_ARGUMENTS} to train.py's argument" | tee -a $LOG_PATH ; }

# change CUDA_VISIABLE_DEVICES according to the machine it runs on
# CUDA_VISIBLE_DEVICES=0,1 \
python3 train.py --max-seq-length $MAX_SEQ_LENGTH \
    --layers-number $LAYERS_NUMBER --attn-heads-number $ATTN_HEADS_NUMBER --embedding-dim $EMBEDDING_DIM \
    --split-ratio $SPLIT_RATIO --batch-size $BATCH_SIZE --steps $STEPS --validation-interval $VALIDATION_INTERVAL --grad-norm-clip $GRAD_NORM_CLIP --early-stop-tolerance $EARLY_STOP_TOLERANCE \
    --lr $LEARNING_RATE --lr-warmup-steps $LEARNING_RATE_WARMUP_STEPS --lr-decay-end-steps $LEARNING_RATE_DECAY_END_STEPS --lr-decay-end-ratio $LEARNING_RATE_DECAY_END_RATIO \
    --use-device $USE_DEVICE --log $LOG_PATH --model-dir-path $MODEL_DIR_PATH $TRAIN_OTHER_ARGUMENTS $CORPUS_DIR_PATH

test $? -ne 0 && { echo "training failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1

echo "pipeline.sh done."
