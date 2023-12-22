#!/bin/bash

if ! { [ $# == 4 ] || [ $# == 5 ]; }; then
    echo "Expect arguments to be:"
    echo "- A configuration file name under configs/model"
    echo "- Two paths to the model directory and the log file"
    echo "- A optional torch device specification"
    exit 1
fi

config_file_paths=()
config_file_paths+=("configs/corpus/${1}.sh")
config_file_paths+=("configs/model/${2}.sh")
model_dir_path=$3
log_path=$4
use_device=$5

for config_file_path in "${config_file_paths[@]}"; do
    if [ -f "$config_file_path" ]; then
        if source "$config_file_path"; then
            echo "${config_file_path}: success"
        else
            echo "${config_file_path}: fail"
            exit 1
        fi
    else
        echo "'${config_file_path}' file not exists"
        exit 1
    fi
done
python3 evaluate_model_wrapper.py \
    --model-dir-path "$model_dir_path" \
    --midi-to-piece-paras "$EVAL_MIDI_TO_PIECE_PARAS_FILE" \
    --softmax-temperature "$SOFTMAX_TEMPERATURE" \
    --sample-function "$SAMPLE_FUNCTION" \
    --sample-threshold "$SAMPLE_THRESHOLD" \
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
