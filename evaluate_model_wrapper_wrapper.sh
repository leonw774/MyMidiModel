#!/bin/bash
# A wrapper of wrapper

if ! { [ $# == 6 ] || [ $# == 5 ]; }; then
    echo "Expect arguments to be:"
    echo "- Three configuration file name for midi preprocessing, bpe, and model setting"
    echo "- Two paths to the model directory and the log file"
    echo "- A optional torch device specification"
    exit 1
fi

corpus_config_file_path="configs/corpus/"$1".sh"
bpe_config_file_path="configs/bpe/"$2".sh"
train_config_file_path="configs/model/"$3".sh"
model_dir_path=$4
log_path=$5
use_device=$6

# check if all argument is a file and execute them to get their vars
for config_file_path in $corpus_config_file_path $bpe_config_file_path $train_config_file_path
do
    if [ -f $config_file_path ]; then
        if source $config_file_path; then
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
    --model-dir-path "$model_dir_path" --num-workers $PROCESS_WORKERS --midi-to-piece-paras "$EVAL_MIDI_TO_PIECE_PARAS_FILE" \
    --log-path "$log_path" --softmax-temperature $SOFTMAX_TEMPERATURE --sample-function $SAMPLE_FUNCTION --sample-threshold $SAMPLE_THRESHOLD \
    --eval-sample-number "$EVAL_SAMPLE_NUMBER" --seed "$SEED" --use-device $use_device --only-eval-uncond "$ONLY_EVAL_UNCOND" \
    -- "$MIDI_DIR_PATH" "$TEST_PATHLIST" $PRIMER_LENGTH

test "$?" -ne 0 && { echo "Evaluation failed. evaluate_model_wrapper_wrapper.sh exit." | tee -a "$log_path" ; } && exit 1
