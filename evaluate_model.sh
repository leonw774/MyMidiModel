#!/bin/bash

help_text="./evaluate_model.sh \
midi_config_name eval_config_name \
[model_dir_path] [log_path] [use_device]"

midi_config_name=$1
eval_config_name=$2
model_dir_path=$3
log_path=$4
use_device=$5

echo "evaluated_model.sh start." | tee -a "$log_path"

if source "configs/corpus/${midi_config_name}.sh"; then
    echo "configs/corpus/${midi_config_name}.sh: success"
else
    echo "configs/corpus/${midi_config_name}.sh: fail"
    exit 1
fi

if source "configs/eval/${eval_config_name}.sh"; then
    echo "configs/eval/${eval_config_name}.sh: success"
else
    echo "configs/eval/${eval_config_name}.sh: fail"
    exit 1
fi

# # assert file and dir exists
if [ ! -d "$MIDI_DIR_PATH" ]; then
    echo "MIDI_DIR_PATH: $MIDI_DIR_PATH is not a directory." \
        "evaluate_model.py exit." | tee -a "$log_path"
fi
if [ ! -f "$TEST_PATHS_FILE" ]; then
    echo "TEST_PATHS_FILE: $TEST_PATHS_FILE is not a file." \
        "evaluate_model.py exit." | tee -a "$log_path"
    exit 1
fi
if [ -z "$model_dir_path" ] && [ ! -d "$model_dir_path" ]; then
    echo "Directory $model_dir_path does not exist"
fi

# set to default if unset or empty
test -z "$log_path"              && log_path=/dev/null
test -z "$EVAL_WORKER_NUMBER"    && EVAL_WORKER_NUMBER=1
test -z "$PRIMER_MEASURE_LENGTH" && PRIMER_MEASURE_LENGTH=0
test -z "$SAMPLE_FUNCTION"       && SAMPLE_FUNCTION=none
test -z "$SAMPLE_THRESHOLD"      && SAMPLE_THRESHOLD=1.0
test -z "$SOFTMAX_TEMPERATURE"   && SOFTMAX_TEMPERATURE=1.0

# set to empty if unset
model_dir_path="${model_dir_path:=}"
MIDI_TO_PIECE_PARAS="${MIDI_TO_PIECE_PARAS:=}"
SEED="${SEED:=}"
use_device="${use_device:=}"
ONLY_EVAL_UNCOND="${ONLY_EVAL_UNCOND:=}"


echo "MIDI_DIR_PATH=${MIDI_DIR_PATH}
model_dir_path=${model_dir_path}
TEST_PATHS_FILE=${TEST_PATHS_FILE}
EVAL_SAMPLE_NUMBER=${EVAL_SAMPLE_NUMBER}
EVAL_WORKER_NUMBER=${EVAL_WORKER_NUMBER}
PRIMER_MEASURE_LENGTH=${PRIMER_MEASURE_LENGTH}
SAMPLE_FUNCTION=${SAMPLE_FUNCTION}
SAMPLE_THRESHOLD=${SAMPLE_THRESHOLD}
SOFTMAX_TEMPERATURE=${SOFTMAX_TEMPERATURE}
MIDI_TO_PIECE_PARAS=${MIDI_TO_PIECE_PARAS}
SEED=${SEED}
use_device=${use_device}
log_path=${log_path}" | tee -a "$log_path"

test_file_number=$(wc -l < "$TEST_PATHS_FILE")
if [ -n "$EVAL_SAMPLE_NUMBER" ] && [ "$EVAL_SAMPLE_NUMBER" -gt 0 ]; then
    sample_number=$EVAL_SAMPLE_NUMBER
else
    if [ "$test_file_number" == 0 ]; then
        echo "Cannot decide sample number:"\
            "no test files or EVAL_SAMPLE_NUMBER given" | tee -a "$log_path"
        echo "evaluate_model.py exit." | tee -a "$log_path"
        exit 1
    else
        echo "Using the number of test files as sample number." \
            | tee -a "$log_path"
        sample_number=$test_file_number
    fi
fi

if [ "$test_file_number" -gt 0 ]; then
    # Have test files, then get their features
    test_copy_dir_path="${MIDI_DIR_PATH}/test_files_copy"
    test_eval_features_path="${MIDI_DIR_PATH}/test_eval_features.json"
    test_eval_features_primer_path="${MIDI_DIR_PATH}/"
    test_eval_features_primer_path+="test_eval_features_primer"
    test_eval_features_primer_path+="${PRIMER_MEASURE_LENGTH}.json"

    # Get features of dataset if no result file

    if [ -f "$test_eval_features_path" ] \
        && [ -f "$test_eval_features_primer_path" ]; then
        echo "Midi dataset ${MIDI_DIR_PATH}" \
            "already has eval features file." | tee -a "$log_path"
    else
        # Copy test files into test_copy_dir_path
        test -d "$test_copy_dir_path" \
            && rm -r "$test_copy_dir_path"
        mkdir "$test_copy_dir_path"
        while read -r test_midi_path; do
            cp "${MIDI_DIR_PATH}/${test_midi_path}" "$test_copy_dir_path"
        done < "$TEST_PATHS_FILE"

        echo "Get evaluation features of $MIDI_DIR_PATH" | tee -a "$log_path"
        python3 get_eval_features_of_midis.py \
            --seed "$SEED" --midi-to-piece-paras "$MIDI_TO_PIECE_PARAS" \
            --log "$log_path" --worker-number "$EVAL_WORKER_NUMBER" \
            -- "$test_copy_dir_path" \
            || {
                echo "Evaluation failed. pipeline.sh exit." \
                    | tee -a "$log_path"
                exit 1;
            }
        temp_path="${test_copy_dir_path}/eval_features.json"
        mv "$temp_path" "$test_eval_features_path"

        echo "Get evaluation features of $MIDI_DIR_PATH" \
            "without first $PRIMER_MEASURE_LENGTH measures" \
            | tee -a "$log_path"
        python3 get_eval_features_of_midis.py \
            --seed "$SEED" --midi-to-piece-paras "$MIDI_TO_PIECE_PARAS" \
            --log "$log_path" --worker-number "$EVAL_WORKER_NUMBER" \
            --primer-measure-length "$PRIMER_MEASURE_LENGTH" \
            -- "$test_copy_dir_path" \
            || {
                echo "Evaluation failed. pipeline.sh exit." \
                    | tee -a "$log_path"
                exit 1;
            }
        temp_path="${test_copy_dir_path}/eval_features.json"
        mv "$temp_path" "$test_eval_features_primer_path"

        rm -r "$test_copy_dir_path"
    fi
fi

test -z "$model_dir_path" && exit 0 

project_root_TEST_PATHS_FILE="${model_dir_path}/test_paths"
test -f "$project_root_TEST_PATHS_FILE" && rm "$project_root_TEST_PATHS_FILE"
touch "$project_root_TEST_PATHS_FILE"
while read -r test_file_path; do
    echo "${MIDI_DIR_PATH}/${test_file_path}" >> "$project_root_TEST_PATHS_FILE";
done < "$TEST_PATHS_FILE"

eval_samples_dir="${model_dir_path}/eval_samples"
if [ ! -d "$eval_samples_dir" ]; then
    mkdir "$eval_samples_dir"
fi

model_file_path="${model_dir_path}/best_model.pt"

### Evaluate model unconditional generation

has_midis=""
ls "${eval_samples_dir}/uncond/"*.mid > /dev/null 2>&1 && has_midis="true"

if [ -d "${eval_samples_dir}/uncond" ] && [ -n "$has_midis" ]; then
    echo "${eval_samples_dir}/uncond already has midi files."

else
    echo "Generating $sample_number unconditional samples" \
        | tee -a "$log_path"
    mkdir "${eval_samples_dir}/uncond"

    start_time=$SECONDS
    python3 generate_with_model.py \
        --seed "$SEED" --use-device "$use_device" --no-tqdm \
        -n "$sample_number" \
        --softmax-temperature "$SOFTMAX_TEMPERATURE" \
        --sample-function "$SAMPLE_FUNCTION" \
        --sample-threshold "$SAMPLE_THRESHOLD" \
        -- "$model_file_path" "${eval_samples_dir}/uncond/uncond" \
        || {
            echo "Generation failed. evaluate_model.sh exit." | tee -a "$log_path"
            exit 1;
        }
    duration=$(( SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${eval_samples_dir}/uncond" \
    | tee -a "$log_path" 
python3 get_eval_features_of_midis.py \
    --seed "$SEED" --midi-to-piece-paras "$MIDI_TO_PIECE_PARAS" \
    --log "$log_path" --worker-number "$EVAL_WORKER_NUMBER" \
    --reference-file-path "$test_eval_features_path" \
    -- "${eval_samples_dir}/uncond" \
    || {
        echo "Evaluation failed. evaluate_model.sh exit." | tee -a "$log_path"
        exit 1;
    }

### Check if stop here

if [ "$test_file_number" == 0 ]; then
    echo "no test files given, "\
        "instrument-conditioned and primer-continuation are omitted." \
        "evaluate_model.py exit." \
        | tee -a "$log_path"
    exit 0
fi

if [ "$ONLY_EVAL_UNCOND" == true ]; then
    echo "ONLY_EVAL_UNCOND is set and true, "\
        "instrument-conditioned and primer-continuation are omitted." \
        "evaluate_model.py exit." \
        | tee -a "$log_path"
    exit 0
fi

### Evaluate model instrument-conditiond generation

has_midis=""
ls "${eval_samples_dir}/instr_cond/"*.mid > /dev/null 2>&1 && has_midis="true"

if [ -d "${eval_samples_dir}/instr_cond" ] && [ -n "$has_midis"  ]; then
    echo "${eval_samples_dir}/instr_cond already has midi files."

else
    echo "Generating $sample_number instrument-conditioned samples" \
        | tee -a "$log_path"
    mkdir "${eval_samples_dir}/instr_cond"

    start_time=$SECONDS
    python3 generate_with_model.py \
        --seed "$SEED" --use-device "$use_device" --no-tqdm \
        -n "$sample_number" \
        -p "$project_root_TEST_PATHS_FILE" -l 0 \
        --softmax-temperature "$SOFTMAX_TEMPERATURE" \
        --sample-function "$SAMPLE_FUNCTION" \
        --sample-threshold "$SAMPLE_THRESHOLD" \
        -- "$model_file_path" "${eval_samples_dir}/instr_cond/instr_cond" \
        || {
            echo "Generation failed. evaluate_model.sh exit." | tee -a "$log_path"
            exit 1;
        }
    duration=$(( SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${eval_samples_dir}/instr-cond" \
    | tee -a "$log_path"
python3 get_eval_features_of_midis.py \
    --seed "$SEED" --midi-to-piece-paras "$MIDI_TO_PIECE_PARAS" \
    --log "$log_path" --worker-number "$EVAL_WORKER_NUMBER" \
    --reference-file-path "$test_eval_features_path" \
    -- "${eval_samples_dir}/instr_cond" \
    || {
        echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path"
        exit 1;
    }

### Evaluate model prime continuation

has_midis=""
ls "${eval_samples_dir}/primer_cont/"*.mid > /dev/null 2>&1 && has_midis="true"

if [ -d "${eval_samples_dir}/primer_cont" ] && [ -n "$has_midis" ]; then
    echo "${eval_samples_dir}/primer_cont already has midi files."

else
    echo "Generating $sample_number prime-continuation samples" \
        | tee -a "$log_path"
    mkdir "${eval_samples_dir}/primer_cont"

    start_time=$SECONDS
    python3 generate_with_model.py \
        --seed "$SEED" --use-device "$use_device" --no-tqdm \
        -n "$sample_number" \
        -p "$project_root_TEST_PATHS_FILE" -l "$PRIMER_MEASURE_LENGTH" \
        --softmax-temperature "$SOFTMAX_TEMPERATURE" \
        --sample-function "$SAMPLE_FUNCTION" \
        --sample-threshold "$SAMPLE_THRESHOLD" \
        -- "$model_file_path" "${eval_samples_dir}/primer_cont/primer_cont" \
        || {
            echo "Generation failed. evaluate_model.sh exit." | tee -a "$log_path"
            exit 1;
        }
    duration=$(( SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${eval_samples_dir}/primer_cont" \
    | tee -a "$log_path"
python3 get_eval_features_of_midis.py \
    --seed "$SEED" --midi-to-piece-paras "$MIDI_TO_PIECE_PARAS" \
    --log "$log_path" --worker-number "$EVAL_WORKER_NUMBER" \
    --primer-measure-length "$PRIMER_MEASURE_LENGTH" \
    --reference-file-path "$test_eval_features_primer_path" \
    -- "${eval_samples_dir}/primer_cont" \
    || {
        echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path"
        exit 1;
    }

echo "evaluated_model.sh exit." | tee -a "$log_path" 
