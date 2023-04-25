#!/bin/bash
midi_dir_path=$1
eval_sample_number=$2
midi_to_piece_paras=$3
process_workers=$4
primer_length=$5
log_path=$6
model_dir_path=$7
nucleus_threshold=$8
seed=$9
if [ -z "$midi_dir_path" ]; then
    echo "./evaluate_model.sh midi_dir_path eval_sample_number midi_to_piece_paras process_workers primer_length log_path model_dir_path nucleus_threshold seed"
    exit 0
fi

# midi_to_piece_paras can be empty string
test -n "$midi_to_piece_paras" && midi_to_piece_paras_option="--midi-ro-piece-paras ${midi_to_piece_paras}"

# seed can be unset
test -n "$9" && seed_option="--seed $8"

echo "evaluated_model.sh start." | tee -a $log_path 
echo "midi_dir_path=${midi_dir_path}, eval_sample_number=${eval_sample_number}, midi_to_piece_paras=${midi_to_piece_paras}, process_workers=${process_workers}"
echo "primer_length=${primer_length}, log_path=${log_path}, model_dir_path=${model_dir_path}, nucleus_threshold=${nucleus_threshold}, seed_option=${seed_option}"

eval_feature_file_path="${midi_dir_path}/eval_features.json"
eval_pathlist_file_path="${midi_dir_path}/eval_pathlist.txt"
eval_primers_feature_file_path="${midi_dir_path}/eval_features_primer${primer_length}.json"

eval_primers_dir_path="${midi_dir_path}/primers${primer_length}"

# Get features of dataset if no result file

if [ -f "$eval_feature_file_path" ] && [ -f "$eval_pathlist_file_path" ] && [ -f "$eval_primers_feature_file_path" ]; then
    echo "Midi dataset $midi_dir_path already has feature stats file." | tee -a $log_path 
else
    echo "Getting evaluation features of $midi_dir_path" | tee -a $log_path 
    python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $eval_sample_number \
        --workers $process_workers --output-sampled-file-paths $midi_dir_path
    test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1
    # Copy sampled files into eval_primers_dir_path
    test -d $eval_primers_dir_path && rm -r $eval_primers_dir_path
    mkdir "$eval_primers_dir_path"
    while read eval_sample_midi_path; do
        cp "$eval_sample_midi_path" "$eval_primers_dir_path"
    done < $eval_pathlist_file_path
    echo "Getting evaluation features without first $primer_length measures of $midi_dir_path" | tee -a $log_path 
    python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $eval_sample_number \
        --workers $process_workers --primer-measure-length $primer_length $eval_primers_dir_path
    # move eval_features.json back to midi dir root
    mv "${eval_primers_dir_path}/eval_features.json" "$eval_primers_feature_file_path"
    # delete eval_primers_dir_path
    rm -r "$eval_primers_dir_path"
fi

test -z "$model_dir_path" && exit 0 

### Evaluate model unconditional generation

has_midis=""
ls "${model_dir_path}/eval_samples/uncond/"*.mid > /dev/null 2>&1 && has_midis="true"
if [ -d "${model_dir_path}/eval_samples/uncond" ] && [ -n "$has_midis" ]; then
    echo "${model_dir_path}/eval_samples/uncond already has midi files."
else
    echo "Generating $eval_sample_number unconditional samples" | tee -a $log_path 
    mkdir "${model_dir_path}/eval_samples/uncond"
    python3 generate_with_model.py --sample-number $eval_sample_number --nucleus-sampling-threshold $nucleus_threshold --no-tqdm --output-text \
        "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/uncond/uncond"
fi

echo "Get evaluation features of ${model_dir_path}/eval_samples/uncond" | tee -a $log_path 
python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $eval_sample_number \
    --workers $process_workers --reference-file-path "$eval_feature_file_path" "${model_dir_path}/eval_samples/uncond"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

### Evaluate model instrument-conditiond generation

has_midis=""
ls "${model_dir_path}/eval_samples/instr_cond/"*.mid > /dev/null 2>&1 && has_midis="true"
if [ -d "${model_dir_path}/eval_samples/instr_cond" ] && [ -n "$has_midis"  ]; then
    echo "${model_dir_path}/eval_samples/instr_cond already has midi files."
else
    echo "Generating $eval_sample_number instrument-conditioned samples" | tee -a $log_path
    mkdir "${model_dir_path}/eval_samples/instr_cond"
    # Loop each line in eval_pathlist_file_path
    while read eval_sample_midi_path; do
        echo "Primer file: $eval_sample_midi_path"
        primer_name=$(basename "$eval_sample_midi_path" .mid)
        python3 generate_with_model.py -p "$eval_sample_midi_path" -l 0 --nucleus-sampling-threshold $nucleus_threshold --no-tqdm --output-text \
            "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/instr_cond/${primer_name}"
    done < $eval_pathlist_file_path
fi

echo "Get evaluation features of ${model_dir_path}/eval_samples/instr-cond" | tee -a $log_path
python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $eval_sample_number \
    --workers $process_workers --reference-file-path "$eval_feature_file_path" "${model_dir_path}/eval_samples/instr_cond"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

### Evaluate model prime continuation

has_midis=""
ls "${model_dir_path}/eval_samples/primer_cont/"*.mid > /dev/null 2>&1 && has_midis="true"
if [ -d "${model_dir_path}/eval_samples/primer_cont" ] && [ -n "$has_midis" ]; then
    echo "${model_dir_path}/eval_samples/primer_cont already has midi files."
else
    echo "Generating $eval_sample_number prime-continuation samples" | tee -a $log_path
    mkdir "${model_dir_path}/eval_samples/primer_cont"
    # Loop each line in eval_pathlist_file_path
    while read eval_sample_midi_path; do
        echo "Primer file: $eval_sample_midi_path"
        primer_name=$(basename "$eval_sample_midi_path" .mid)
        python3 generate_with_model.py -p "$eval_sample_midi_path" -l $primer_length --nucleus-sampling-threshold $nucleus_threshold --no-tqdm --output-text \
            "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/primer_cont/${primer_name}"
    done < $eval_pathlist_file_path
fi

echo "Get evaluation features of ${model_dir_path}/eval_samples/primer_cont" | tee -a $log_path
python3 get_eval_features_of_midis.py $seed_option $midi_to_piece_paras_option --log $log_path --sample-number $eval_sample_number \
    --primer-measure-length $primer_length \
    --workers $process_workers --reference-file-path "$eval_primers_feature_file_path" "${model_dir_path}/eval_samples/primer_cont"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

echo "evaluated_model.sh exit." | tee -a $log_path 
