#!/bin/bash

help_text="var1=value1 var2=value2 [...] varN=valueN ./evaluate_model.sh
Required variables:
\tmidi_dir_path test_paths_file primer_measure_length
Optional variables:
\tmodel_dir_path worker_number log_path midi_to_piece_paras\
softmax_temperature sample_function sample_threshold seed
"
# check required variables
if [ -z "$midi_dir_path" ] \
    || [ -z "$test_paths_file" ] \
    || [ -z "$primer_measure_length" ] ; then
    printf "%s" "$help_text"
    exit 1
fi

if [ ! -d "$midi_dir_path" ]; then
    echo "midi_dir_path: $midi_dir_path is not a directory." \
        "evaluate_model.py exit." | tee -a "$log_path"
    exit 1
fi

if [ ! -f "$test_paths_file" ]; then
    echo "test_paths_file: $test_paths_file is not a file." \
        "evaluate_model.py exit." | tee -a "$log_path"
    exit 1
fi

# optionals: set default and make option string
test -z "$log_path"            && log_path=/dev/null
test -z "$worker_number"       && worker_number=1
test -z "$softmax_temperature" && softmax_temperature=1.0
test -z "$sample_function"     && sample_function=none
test -z "$sample_threshold"    && sample_threshold=1.0

# set to empty string if unset
model_dir_path="${model_dir_path:=}"
midi_to_piece_paras="${midi_to_piece_paras:=}"
seed="${seed:=}"
use_device="${use_device:=}"
only_eval_uncond="${only_eval_uncond:=}"

echo "evaluated_model.sh start." | tee -a "$log_path"
echo "midi_dir_path=${midi_dir_path}
model_dir_path=${model_dir_path}
test_paths_file=${test_paths_file}
primer_measure_length=${primer_measure_length}
eval_sample_number=${eval_sample_number}
worker_number=${worker_number}
midi_to_piece_paras=${midi_to_piece_paras}
softmax_temperature=${softmax_temperature}
sample_function=${sample_function}
sample_threshold=${sample_threshold}
seed=${seed}
use_device=${use_device}
log_path=${log_path}" | tee -a "$log_path"

test_file_number=$(wc -l < "$test_paths_file")
if [ -n "$eval_sample_number" ] && [ "$eval_sample_number" -gt 0 ]; then
    sample_number=$eval_sample_number
else
    if [ "$test_file_number" == 0 ]; then
        echo "Cannot decide sample number:"\
            "no test files or eval_sample_number given" | tee -a "$log_path"
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
    test_copy_dir_path="${midi_dir_path}/test_files_copy"
    test_eval_features_path="${midi_dir_path}/test_eval_features.json"
    test_eval_features_primer_path="${midi_dir_path}/"
    test_eval_features_primer_path+="test_eval_features_primer.json"

    # Get features of dataset if no result file

    if [ -f "$test_eval_features_path" ] \
        && [ -f "$test_eval_features_primer_path" ]; then
        echo "Midi dataset ${midi_dir_path}" \
            "already has eval features file." | tee -a "$log_path"
    else
        # Copy test files into test_copy_dir_path
        test -d "$test_copy_dir_path" \
            && rm -r "$test_copy_dir_path"
        mkdir "$test_copy_dir_path"
        while read -r test_midi_path; do
            cp "${midi_dir_path}/${test_midi_path}" "$test_copy_dir_path"
        done < "$test_paths_file"

        echo "Get evaluation features of $midi_dir_path" | tee -a "$log_path"
        python3 get_eval_features_of_midis.py \
            --seed "$seed" --midi-to-piece-paras "$midi_to_piece_paras" \
            --log "$log_path" --worker-number "$worker_number" \
            -- "$test_copy_dir_path" \
            || {
                echo "Evaluation failed. pipeline.sh exit." \
                    | tee -a "$log_path"
                exit 1;
            }
        temp_path="${test_copy_dir_path}/eval_features.json"
        mv "$temp_path" "$test_eval_features_path"

        echo "Get evaluation features of $midi_dir_path" \
            "without first $primer_measure_length measures" \
            | tee -a "$log_path"
        python3 get_eval_features_of_midis.py \
            --seed "$seed" --midi-to-piece-paras "$midi_to_piece_paras" \
            --log "$log_path" --worker-number "$worker_number" \
            --primer-measure-length "$primer_measure_length" \
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

project_root_test_paths_file="${model_dir_path}/test_paths"
test -f "$project_root_test_paths_file" && rm "$project_root_test_paths_file"
touch "$project_root_test_paths_file"
while read -r test_file_path; do
    echo "${midi_dir_path}/${test_file_path}" >> "$project_root_test_paths_file";
done < "$test_paths_file"

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
        --seed "$seed" --use-device "$use_device" --no-tqdm \
        -n "$sample_number" \
        --softmax-temperature "$softmax_temperature" \
        --sample-function "$sample_function" \
        --sample-threshold "$sample_threshold" \
        --sample-threshold-head-multiplier "$SAMPLE_THRESHOLD_HEAD_MULTIPLIER" \
        -- "$model_file_path" "${eval_samples_dir}/uncond/uncond"
    duration=$(( SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${eval_samples_dir}/uncond" \
    | tee -a "$log_path" 
python3 get_eval_features_of_midis.py \
    --seed "$seed" --midi-to-piece-paras "$midi_to_piece_paras" \
    --log "$log_path" --worker-number "$worker_number" \
    --reference-file-path "$test_eval_features_path" \
    -- "${eval_samples_dir}/uncond" \
    || {
        echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path"
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

if [ "$only_eval_uncond" == true ]; then
    echo "only_eval_uncond is set and true, "\
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
        --seed "$seed" --use-device "$use_device" --no-tqdm \
        -n "$sample_number" \
        -p "$project_root_test_paths_file" -l 0 \
        --softmax-temperature "$softmax_temperature" \
        --sample-function "$sample_function" \
        --sample-threshold "$sample_threshold" \
        --sample-threshold-head-multiplier "$SAMPLE_THRESHOLD_HEAD_MULTIPLIER" \
        -- "$model_file_path" "${eval_samples_dir}/instr_cond/instr_cond"
    duration=$(( SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${eval_samples_dir}/instr-cond" \
    | tee -a "$log_path"
python3 get_eval_features_of_midis.py \
    --seed "$seed" --midi-to-piece-paras "$midi_to_piece_paras" \
    --log "$log_path" --worker-number "$worker_number" \
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
        --seed "$seed" --use-device "$use_device" --no-tqdm \
        -n "$sample_number" \
        -p "$project_root_test_paths_file" -l "$primer_measure_length" \
        --softmax-temperature "$softmax_temperature" \
        --sample-function "$sample_function" \
        --sample-threshold "$sample_threshold" \
        --sample-threshold-head-multiplier "$SAMPLE_THRESHOLD_HEAD_MULTIPLIER" \
        -- "$model_file_path" "${eval_samples_dir}/primer_cont/primer_cont"
    duration=$(( SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${eval_samples_dir}/primer_cont" \
    | tee -a "$log_path"
python3 get_eval_features_of_midis.py \
    --seed "$seed" --midi-to-piece-paras "$midi_to_piece_paras" \
    --log "$log_path" --worker-number "$worker_number" \
    --primer-measure-length "$primer_measure_length" \
    --reference-file-path "$test_eval_features_primer_path" \
    -- "${eval_samples_dir}/primer_cont" \
    || {
        echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path"
        exit 1;
    }

echo "evaluated_model.sh exit." | tee -a "$log_path" 
