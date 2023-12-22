#!/bin/bash

if ! { [ $# == 6 ] || [ $# == 5 ]; }; then
    echo "Expect arguments to be:"
    echo "- A configuration file name under configs/model"
    echo "- Two paths to the model directory and the log file"
    echo "- A optional torch device specification"
    exit 1
fi

model_config_file_paths="configs/model/${1}.sh"
model_dir_path=$2
log_path=$3
use_device=$4

if [ -f "$model_config_file_paths" ]; then
    if source "$model_config_file_paths"; then
        echo "source ${model_config_file_paths}: success"
    else
        echo "source ${model_config_file_paths}: fail"
        exit 1
    fi
else
    echo "'${model_config_file_paths}' file not exists"
    exit 1
fi

python3 evaluate_model_wrapper.py \
    --model-dir-path "$model_dir_path" \
    --midi-to-piece-paras "$EVAL_MIDI_TO_PIECE_PARAS_FILE" \
    --softmax-temperature "$SOFTMAX_TEMPERATURE" \
    --sample-function "$SAMPLE_FUNCTION" \
    --sample-threshold "$SAMPLE_THRESHOLD" \
    --sample-threshold-head-multiplier "$SAMPLE_THRESHOLD_HEAD_MULTIPLIER" \
    --eval-sample-number "$EVAL_SAMPLE_NUMBER" \
    --worker-number "$EVAL_WORKER_NUMBER" \
    --seed "$SEED" --log-path "$log_path" \
    --only-eval-uncond "$ONLY_EVAL_UNCOND" \
    -- "$MIDI_DIR_PATH" "$TEST_PATHS_FILE" "$PRIMER_LENGTH" \
    || {
        echo "Evaluation failed. evaluate_model_wrapper_wrapper.sh exit." \
            | tee -a "$log_path"
        exit 1
    }
