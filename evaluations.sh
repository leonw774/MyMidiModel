midi_dir_path=$1
eval_sample_number=$2
process_workers=$3
primer_length=$4
log_path=$5
model_dir_path=$6
nucleus_threshold=$7

eval_primers_dir_path="${midi_dir_path}/eval_primers_${eval_sample_number}"
eval_primers_pathlist_file_path="${midi_dir_path}/eval_pathlist.txt"

# Get features of dataset if no result file

if [ -f "${midi_dir_path}/eval_features.json" ] && [ -f "$eval_primers_pathlist_file_path" ] && [ -d "$eval_primers_dir_path" ]; then
    echo "Midi dataset $midi_dir_path already has feature stats file." | tee -a $log_path 
else
    echo "Getting evaluation features of $midi_dir_path" | tee -a $log_path 
    python3 get_eval_features_of_midis.py --log $log_path --sample-number $eval_sample_number --workers $process_workers \
        --output-sampled-file-paths $midi_dir_path
    test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1
    # Copy sampled files into eval_primers_dir_path
    test -d $eval_primers_dir_path && rm -r $eval_primers_dir_path
    mkdir $eval_primers_dir_path
    while read eval_sample_midi_path; do
        cp $eval_sample_midi_path $eval_primers_dir_path
    done < $eval_primers_pathlist_file_path
    echo "Getting evaluation features without first $primer_length measures of $midi_dir_path" | tee -a $log_path 
    python3 get_eval_features_of_midis.py --log $log_path --sample-number $eval_sample_number --workers $process_workers \
        --primer-measure-length $primer_length $eval_primers_dir_path
fi

test -z "$model_dir_path" && exit 0 

### Evaluate model unconditional generation

echo "Generating $eval_sample_number unconditional samples" | tee -a $log_path 
mkdir "${model_dir_path}/eval_samples/uncond"
python3 generate_with_model.py --sample-number $eval_sample_number --nucleus-sampling-threshold $nucleus_threshold --no-tqdm --output-txt \
    "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/uncond/sample"

echo "Get evaluation features of ${model_dir_path}/eval_samples/uncond" | tee -a $log_path 
python3 get_eval_features_of_midis.py --log $log_path --sample-number $eval_sample_number --workers $process_workers \
    --reference-file-path "${midi_dir_path}/eval_features.json" "${model_dir_path}/eval_samples/uncond"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

### Evaluate model instrument-conditiond generation

echo "Generating $eval_sample_number instrument-conditioned samples" | tee -a $log_path
mkdir "${model_dir_path}/eval_samples/instr_cond"
# Loop each line in eval_primers_pathlist_file_path
while read eval_sample_midi_path; do
    echo "Primer file: $eval_sample_midi_path"
    python3 generate_with_model.py -p $eval_sample_midi_path -l 0 --nucleus-sampling-threshold $nucleus_threshold --no-tqdm --output-txt \
    "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/instr_cond/$(basename $eval_sample_midi_path .mid)"
done < $eval_primers_pathlist_file_path

echo "Get evaluation features of ${model_dir_path}/eval_samples/instr-cond" | tee -a $log_path
python3 get_eval_features_of_midis.py --log $log_path --sample-number $eval_sample_number --workers $process_workers \
    --reference-file-path "${midi_dir_path}/eval_features.json" "${model_dir_path}/eval_samples/instr_cond"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

### Evaluate model prime continuation

echo "Generating $eval_sample_number prime-continuation samples" | tee -a $log_path
mkdir "${model_dir_path}/eval_samples/primer_cont"
# Loop each line in eval_primers_pathlist_file_path
while read eval_sample_midi_path; do
    echo "Primer file: $eval_sample_midi_path"
    python3 generate_with_model.py -p $eval_sample_midi_path -l $primer_length --nucleus-sampling-threshold $nucleus_threshold --no-tqdm --output-txt \
    "${model_dir_path}/best_model.pt" "${model_dir_path}/eval_samples/primer_cont/$(basename $eval_sample_midi_path .mid)"
done < $eval_primers_pathlist_file_path

echo "Get evaluation features of ${model_dir_path}/eval_samples/primer_cont" | tee -a $log_path
python3 get_eval_features_of_midis.py --log $log_path --sample-number $eval_sample_number --workers $process_workers \
    --primer-measure-length $primer_length --reference-file-path "${eval_primers_dir_path}/eval_features.json" "${model_dir_path}/eval_samples/primer_cont"
test $? -ne 0 && { echo "Evaluation failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1
