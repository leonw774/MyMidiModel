#!/bin/bash

# check required variables
if [ -z "$midi_dir_path" ] || [ -z "$test_pathlist" ] || [ -z "$primer_measure_length" ] ; then
    echo "var1=value1 var2=value2 [...] varN=valueN ./evaluate_model.sh"
    echo "Required variables:"
    echo -e "\tmidi_dir_path test_pathlist primer_measure_length"
    echo "Optional variables:"
    echo -e "\tmodel_dir_path num_workers log_path midi_to_piece_paras softmax_temperature nucleus_sampling_threshold seed"
    exit 1
fi

# optionals: set default and make option string
test -z "$log_path" && log_path=/dev/null
test -z "$num_workers" && num_workers=1
test -z "$softmax_temperature" && softmax_temperature=1.0
test -z "$nucleus_sampling_threshold" && nucleus_sampling_threshold=1.0

test -n "$seed" && seed_option="--seed $seed"
test -n "$midi_to_piece_paras" && midi_to_piece_paras_option="--midi-to-piece-paras ${midi_to_piece_paras}"

sample_number=$(wc -l < $test_pathlist)

echo "evaluated_model.sh start." | tee -a "$log_path"
echo "midi_dir_path=${midi_dir_path} test_pathlist=${test_pathlist} primer_measure_length=${primer_measure_length} sample_number=${sample_number}" | tee -a "$log_path"
echo "model_dir_path=${model_dir_path} midi_to_piece_paras_option=${midi_to_piece_paras_option} seed_option=${seed_option}" | tee -a "$log_path"
echo "num_workers=${num_workers} log_path=${log_path} softmax_temperature=${softmax_temperature} nucleus_sampling_threshold=${nucleus_sampling_threshold}" | tee -a "$log_path"

if [ "$sample_number" == 0 ]; then
    echo "No test files"
else
    test_eval_feature_path="${midi_dir_path}/eval_features.json"
    test_primers_eval_feature_path="${midi_dir_path}/eval_features_primer${primer_measure_length}.json"
    primers_dir_path="${midi_dir_path}/primers${primer_measure_length}"

    # Get features of dataset if no result file

    if [ -f "$test_eval_feature_path" ] && [ -f "$test_primers_eval_feature_path" ]; then
        echo "Midi dataset $midi_dir_path already has feature stats file." | tee -a "$log_path" 
    else
        # Copy test files into primers_dir_path
        test -d $primers_dir_path && rm -r $primers_dir_path
        mkdir "$primers_dir_path"
        while read test_midi_path; do
            cp "${midi_dir_path}/${test_midi_path}" "$primers_dir_path"
        done < $test_pathlist

        echo "Getting evaluation features of $midi_dir_path" | tee -a "$log_path" 
        python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path \
            --workers $num_workers -- $primers_dir_path
        test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path" ; } && exit 1
        mv "${primers_dir_path}/eval_features.json" "$test_eval_feature_path"

        echo "Getting evaluation features without first $primer_measure_length measures" | tee -a "$log_path" 
        python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path \
            --workers $num_workers --primer-measure-length $primer_measure_length -- $primers_dir_path
        # move eval_features.json back to midi dir root
        mv "${primers_dir_path}/eval_features.json" "$test_primers_eval_feature_path"
        # delete primers_dir_path
        # rm -r "$primers_dir_path"
    fi
fi

test -z "$model_dir_path" && exit 0 

### Evaluate model unconditional generation

has_midis=""
ls "${model_dir_path}/eval_samples/uncond/"*.mid > /dev/null 2>&1 && has_midis="true"
if [ -d "${model_dir_path}/eval_samples/uncond" ] && [ -n "$has_midis" ]; then
    echo "${model_dir_path}/eval_samples/uncond already has midi files."
else
    echo "Generating $sample_number unconditional samples" | tee -a "$log_path"
    start_time=$SECONDS
    mkdir "${model_dir_path}/eval_samples/uncond"
    python3 generate_with_model.py $seed_option --sample-number $sample_number --no-tqdm \
        --softmax-temperature $softmax_temperature --nucleus-sampling-threshold $nucleus_sampling_threshold -- \
        "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/uncond/uncond"
    duration=$(( $SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${model_dir_path}/eval_samples/uncond" | tee -a "$log_path" 
python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $sample_number \
    --workers $num_workers --reference-file-path "$test_eval_feature_path" "${model_dir_path}/eval_samples/uncond"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path" ; } && exit 1

### Evaluate model instrument-conditiond generation

has_midis=""
ls "${model_dir_path}/eval_samples/instr_cond/"*.mid > /dev/null 2>&1 && has_midis="true"
if [ -d "${model_dir_path}/eval_samples/instr_cond" ] && [ -n "$has_midis"  ]; then
    echo "${model_dir_path}/eval_samples/instr_cond already has midi files."
else
    echo "Generating $sample_number instrument-conditioned samples" | tee -a "$log_path"
    mkdir "${model_dir_path}/eval_samples/instr_cond"
    start_time=$SECONDS
    # Loop each line in test_pathlist
    while read test_midi_path; do
        complete_test_midi_path="${midi_dir_path}/${test_midi_path}"
        echo "Primer file: $complete_test_midi_path"
        primer_name=$(basename "$complete_test_midi_path" .mid)
        python3 generate_with_model.py $seed_option -p "$complete_test_midi_path" -l 0 --no-tqdm \
            --softmax-temperature $softmax_temperature --nucleus-sampling-threshold $nucleus_sampling_threshold -- \
            "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/instr_cond/${primer_name}"
    done < $test_pathlist
    duration=$(( $SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${model_dir_path}/eval_samples/instr-cond" | tee -a "$log_path"
python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $sample_number \
    --workers $num_workers --reference-file-path "$test_eval_feature_path" "${model_dir_path}/eval_samples/instr_cond"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path" ; } && exit 1

### Evaluate model prime continuation

has_midis=""
ls "${model_dir_path}/eval_samples/primer_cont/"*.mid > /dev/null 2>&1 && has_midis="true"
if [ -d "${model_dir_path}/eval_samples/primer_cont" ] && [ -n "$has_midis" ]; then
    echo "${model_dir_path}/eval_samples/primer_cont already has midi files."
else
    echo "Generating $sample_number prime-continuation samples" | tee -a "$log_path"
    mkdir "${model_dir_path}/eval_samples/primer_cont"
    start_time=$SECONDS
    # Loop each line in test_pathlist
    while read test_midi_path; do
        complete_test_midi_path="${midi_dir_path}/${test_midi_path}"
        echo "Primer file: $complete_test_midi_path"
        primer_name=$(basename "$complete_test_midi_path" .mid)
        python3 generate_with_model.py $seed_option -p "$complete_test_midi_path" -l $primer_measure_length --no-tqdm \
            --softmax-temperature $softmax_temperature --nucleus-sampling-threshold $nucleus_sampling_threshold -- \
            "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/primer_cont/${primer_name}"
    done < $test_pathlist
    duration=$(( $SECONDS - start_time ))
    echo "Finished. Used time: ${duration} seconds" | tee -a "$log_path"
fi

echo "Get evaluation features of ${model_dir_path}/eval_samples/primer_cont" | tee -a "$log_path"
python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $sample_number \
    --primer-measure-length $primer_measure_length \
    --workers $num_workers --reference-file-path "$test_primers_eval_feature_path" "${model_dir_path}/eval_samples/primer_cont"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path" ; } && exit 1

echo "evaluated_model.sh exit." | tee -a "$log_path" 
