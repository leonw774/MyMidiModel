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

DO_MIDI_TO_TEXT=true
DO_BPE=false
if [ $BPE_ITER -ne 0 ]; then
    DO_BPE=true
    CORPUS_DIR_PATH_WITH_BPE="${CORPUS_DIR_PATH}_bpe${BPE_ITER}_${SCORE_FUNC}_${MERGE_CONDITION}_${SAMPLE_RATE}"
    if [ -d $CORPUS_DIR_PATH_WITH_BPE ] && [ -f "${CORPUS_DIR_PATH_WITH_BPE}/corpus" ] && [ -f "${CORPUS_DIR_PATH_WITH_BPE}/shape_vocab" ]; then
        if [ -n "${USE_EXISTED}" ]; then
            echo "BPE Output directory: ${CORPUS_DIR_PATH_WITH_BPE} already has corpus and shape_vocab file." | tee -a $LOG_PATH
            echo "Flag --use-existed is set" | tee -a $LOG_PATH
            echo "Learn bpe vocab is skipped" | tee -a $LOG_PATH
            DO_BPE=false
            DO_MIDI_TO_TEXT=false
        else
            echo "BPE Output directory: ${CORPUS_DIR_PATH_WITH_BPE} already has corpus and shape_vocab file. Remove? (y=remove/n=skip bpe)" | tee -a $LOG_PATH
            read yn
            if [ "$yn" == "${yn#[Yy]}" ]; then 
            # this grammar (the #[] operator) means that, in the variable $yn, any Y or y in 1st position will be dropped if they exist.
                # enter this block if yn != [Yy]
                DO_BPE=false
                DO_MIDI_TO_TEXT=false
                echo "Learn bpe vocab is skipped" | tee -a $LOG_PATH
            else
                rm -f "${CORPUS_DIR_PATH_WITH_BPE}/*"
            fi
        fi
    fi
fi

if [ "$DO_MIDI_TO_TEXT" == true ]; then
    MIDI_TO_TEXT_OTHER_ARGUMENTS=""
    test "$CONTINUING_NOTE" == true && MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-continuing-note"
    test "$USE_MERGE_DRUMS" == true && MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --use-merge-drums"
    test "$MIDI_TO_TEXT_VERBOSE" == true && MIDI_TO_TEXT_OTHER_ARGUMENTS="${MIDI_TO_TEXT_OTHER_ARGUMENTS} --verbose"
    test -n "$TRAIN_OTHER_ARGUMENTS" && { echo "Appended ${MIDI_TO_TEXT_OTHER_ARGUMENTS} to midi_to_text's argument" | tee -a $LOG_PATH ; }

    echo "Corpus dir: ${CORPUS_DIR_PATH}"

    python3 midi_to_corpus.py --nth $NTH --max-track-number $MAX_TRACK_NUMBER --max-duration $MAX_DURATION --velocity-step $VELOCITY_STEP \
        --tempo-quantization $TEMPO_MIN $TEMPO_MAX $TEMPO_STEP --position-method $POSITION_METHOD $MIDI_TO_TEXT_OTHER_ARGUMENTS $USE_EXISTED \
        --log $LOG_PATH -w $PROCESS_WORKERS -r -o $CORPUS_DIR_PATH $MIDI_DIR_PATH
    test $? -ne 0 && { echo "midi_to_text.py failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1
fi

if [ "$DO_BPE" == true ]; then
    echo "Start learn bpe vocab" | tee -a $LOG_PATH
    # compile
    make -C ./bpe
    test $? -ne 0 && { echo "Compile error. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1

    # create new dir 
    if [ -d $CORPUS_DIR_PATH_WITH_BPE ]; then
        rm -f "${CORPUS_DIR_PATH_WITH_BPE}/*"
    else
        mkdir $CORPUS_DIR_PATH_WITH_BPE
    fi

    # copy paras and pathlist
    cp "${CORPUS_DIR_PATH}/paras" $CORPUS_DIR_PATH_WITH_BPE
    cp "${CORPUS_DIR_PATH}/pathlist" $CORPUS_DIR_PATH_WITH_BPE

    # run learn_vocab
    bpe/learn_vocab $BPE_DOLOG $BPE_CLEARLINE $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE $BPE_ITER $SCORE_FUNC $MERGE_CONDITION $SAMPLE_RATE $MIN_SCORE_LIMIT | tee -a $LOG_PATH
    
    BPE_EXIT_CODE=${PIPESTATUS[0]}
    if [ $BPE_EXIT_CODE -ne 0 ]; then
        echo "learn_vocab failed. exit code: $BPE_EXIT_CODE. pipeline.sh exit." | tee -a $LOG_PATH
        echo "bpe/learn_vocab $BPE_DOLOG $BPE_CLEARLINE $CORPUS_DIR_PATH $CORPUS_DIR_PATH_WITH_BPE $BPE_ITER $SCORE_FUNC $MERGE_CONDITION $SAMPLE_RATE $MIN_SCORE_LIMIT"
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
    test $VERIFY_EXIT_CODE -ne 0 && { echo "Corpus equality verification failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1
fi

if [ $BPE_ITER -ne 0 ]; then
    # replace CORPUS_DIR_PATH to CORPUS_DIR_PATH_WITH_BPE
    CORPUS_DIR_PATH=$CORPUS_DIR_PATH_WITH_BPE
fi

python3 make_arrays.py --debug --bpe $BPE_ITER --mp-worker-number $PROCESS_WORKERS --log $LOG_PATH $CORPUS_DIR_PATH $USE_EXISTED
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
if [ "$USE_PERMUTABLE_SUBSEQ_LOSS" == true ]; then
    python3 -c "import torch;import mps_loss" 2>&1 | grep ModuleNotFoundError
    # if module not exist
    if [ $? -eq 0 ]; then
        cd util/pytorch/mps_loss
        python3 setup.py install
        cd ../../..
    fi
    # check if success
    python3 -c "import torch;import mps_loss" 2>&1 | grep ModuleNotFoundError
    test $? -eq 0 && { echo "mps_loss extension setup failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-permutable-subseq-loss"
fi
test "$PERMUTE_MPS" == true                && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-mps"
test "$PERMUTE_TRACK_NUMBER" == true       && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-track-number"
test "$USE_LINEAR_ATTENTION" == true       && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-linear-attn"
test "$INPUT_NO_TEMPO" == true             && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --input-no-tempo"
test "$INPUT_NO_TIME_SIGNATURE" == true    && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --input-no-time-signatrue"
test "$USE_PARALLEL" == true               && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-parallel"
test -n "$MAX_PIECE_PER_GPU"               && TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --max-pieces-per-gpu ${MAX_PIECE_PER_GPU}"
test -n "$TRAIN_OTHER_ARGUMENTS" && { echo "Appended${TRAIN_OTHER_ARGUMENTS} to train.py's argument" | tee -a $LOG_PATH ; }


# change CUDA_VISIABLE_DEVICES according to the machine it runs on
if [ "$USE_PARALLEL" == true ]; then
    NUM_OF_CUDA_VISIBLE_DEVICE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c ;)
    NUM_OF_CUDA_VISIBLE_DEVICE=$(($NUM_OF_CUDA_VISIBLE_DEVICE+1)) # perform arithmetic expression with $((...))
    LAUNCH_COMMAND="accelerate launch --multi_gpu --num_processes $NUM_OF_CUDA_VISIBLE_DEVICE --num_machine 1"
    accelerate config default
else
    LAUNCH_COMMAND="python3"
fi
$LAUNCH_COMMAND train.py --max-seq-length $MAX_SEQ_LENGTH --pitch-augmentation $PITCH_AUGMENTATION $TRAIN_OTHER_ARGUMENTS \
    --layers-number $LAYERS_NUMBER --attn-heads-number $ATTN_HEADS_NUMBER --embedding-dim $EMBEDDING_DIM \
    --batch-size $BATCH_SIZE --max-steps $MAX_STEPS --grad-norm-clip $GRAD_NORM_CLIP \
    --split-ratio $SPLIT_RATIO --validation-interval $VALIDATION_INTERVAL --validation-steps $VALIDATION_STEPS --early-stop $EARLY_STOP \
    --lr-peak $LEARNING_RATE_PEAK --lr-warmup-steps $LEARNING_RATE_WARMUP_STEPS --lr-decay-end-steps $LEARNING_RATE_DECAY_END_STEPS --lr-decay-end-ratio $LEARNING_RATE_DECAY_END_RATIO \
    --use-device $USE_DEVICE --log $LOG_PATH $CORPUS_DIR_PATH $MODEL_DIR_PATH

test $? -ne 0 && { echo "training failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1

python3 get_eval_features_of_model.py --log $LOG_PATH $MODEL_DIR_PATH

test $? -ne 0 && { echo "evaluation failed. pipeline.sh exit." | tee -a $LOG_PATH ; } && exit 1

if [ -n "$USE_EXISTED" ] && [ -f "${MIDI_DIR_PATH}/eval_sample_feature_stats.json" ] ; then
    echo "Midi dataset ${MIDI_DIR_PATH} already has feature stats file. get_eval_features_of_dataset.py is skipped."
else
    python3 get_eval_features_of_dataset.py --log $LOG_PATH $MIDI_DIR_PATH
fi

echo "pipeline.sh done."
