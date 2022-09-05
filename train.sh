echo "train.sh start."
if [ $# -ne 1 ]; then
    echo "Expect arguments to be three configuration file name for midi preprocessing, bpe setting and training/model setting."
fi

CONFIG_PATH="configs/train/"$1".sh"
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

TRAIN_NAME=$CORPUS_DIR_PATH"-"$1
LOG_PATH="logs/$(date '+%Y%m%d%H%M')-"$TRAIN_NAME
echo "Log file: $LOG_PATH"
touch $LOG_PATH

# make directory for checkpoints and loss csv
CHECKPOINT_DIR_PATH="ckpt/$(date '+%Y%m%d%H%M')-"$TRAIN_NAME
mkdir CHECKPOINT_DIR_PATH

TRAIN_OTHER_ARGUMENTS=""
if [ $USE_PERMUTABL_SUBSEQ_LOSS == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --use-permutable-subseq-loss"
fi
if [ $PERMUTE_MPS == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-mps"
fi
if [ $PERMUTE_TRACK_NUMBER == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --permute-track-number"
fi
if [ $EARLY-STOP == true ]; then
    TRAIN_OTHER_ARGUMENTS="${TRAIN_OTHER_ARGUMENTS} --early-stop"
fi

python3 train.py --max-seq-length $MAX_SEQ_LENGTH --sample-stride $SAMPLE_STRIDE \
    --layers-number $LAYERS_NUMBER --attn-heads-number $ATTN_HEADS_NUMBER --embedding-dim $EMBEDDING_DIM \
    --split-ratio $SPLIT_RATIO --batch-size $BATCH_SIZE --steps $STEPS --validation-interval $VALIDATION_INTERVAL --grad-norm-clip $GRAD_NORM_CLIP --early-stop-tolerance $EARLY_STOP_TOLERANCE $TRAIN_OTHER_ARGUMENTS \
    --lr $LEARNING_RATE --lr-warmup-steps $LEARNING_RATE_WARMUP_STEPS --lr-decay-end-steps $LEARNING_RATE_DECAY_END_STEPS -lr-decay-end-ratio $LEARNING_RATE_DECAY_END_RATIO \
    --use-device $USE_DEVICE --log $LOG_PATH --checkpoint-dir-path $CHECKPOINT_DIR_PATH $CORPUS_DIR_PATH
    