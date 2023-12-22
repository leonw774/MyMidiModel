#!/bin/bash

if ! { [ $# == 6 ] || [ $# == 5 ]; }; then
    echo "Expect arguments to be:"
    echo "- Three configuration file name for midi preproc, bpe, and model"
    echo "- Two paths to the model directory and the log file"
    echo "- A optional torch device specification"
    exit 1
fi

config_file_paths=()
config_file_paths+=("configs/corpus/${1}.sh")
config_file_paths+=("configs/bpe/${2}.sh")
config_file_paths+=("configs/model/${3}.sh")

model_dir_path=$4
log_path=$5
use_device=$6

# check if all argument is a file and execute them to get their vars
for config_file_path in "${config_file_paths[@]}"; do
    if [ -f "$config_file_path" ]; then
        if source "$config_file_path"; then
            echo "source ${config_file_path}: success"
        else
            echo "source ${config_file_path}: fail"
            exit 1
        fi
    else
        echo "'${config_file_path}' file not exists"
        exit 1
    fi
done

python3 evaluate_model_wrapper.py \
    --model-dir-path "$model_dir_path" --worker-number "$WORKER_NUMBER" \
    --midi-to-piece-paras "$EVAL_MIDI_TO_PIECE_PARAS_FILE" \
    --softmax-temperature "$SOFTMAX_TEMPERATURE" \
    --sample-function "$SAMPLE_FUNCTION" --sample-threshold "$SAMPLE_THRESHOLD" \
    --eval-sample-number "$EVAL_SAMPLE_NUMBER" --seed "$SEED" \
    --log-path "$log_path" --only-eval-uncond "$ONLY_EVAL_UNCOND" \
    --use-device "$use_device" \
    -- "$MIDI_DIR_PATH" "$TEST_PATHS_FILE" "$PRIMER_LENGTH" \
    || {
        echo "Evaluation failed. evaluate_model_wrapper_wrapper.sh exit." \
            | tee -a "$log_path"
        exit 1
    }
