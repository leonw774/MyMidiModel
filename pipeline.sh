#!/bin/bash
echo "pipeline.sh start."
use_existed_flag=""
if [ $# -eq 4 ]; then
    if [ "$4" == '--use-existed' ]; then
        use_existed_flag="--use-existed"
    else
        echo "Expect arguments to be three configuration file name for \
            midi preprocessing, bpe, and model/eval setting, \
            and an optional '--use-existed' flag at the fourth position."
        exit 1
    fi
else
    if [ $# -ne 3 ]; then
        echo "Expect arguments to be three configuration file name for \
            midi preprocessing, bpe, and model/eval setting, \
            and an optional '--use-existed' flag at the fourth position."
        exit 1
    fi
fi

# check if all argument is a file and execute them to get their vars
full_config_name="${1}-${2}-${3}"
config_file_paths=()
config_file_paths+=("configs/corpus/${1}.sh")
config_file_paths+=("configs/bpe/${2}.sh")
config_file_paths+=("configs/model/${3}.sh")

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

log_path="logs/$(date '+%Y%m%d-%H%M%S')-${full_config_name}.log"
echo "Log file: $log_path"
touch "$log_path"

######## MAKE CORPUS ########

corpus_path_flags=""
test "$CONTINUING_NOTE" != true && corpus_path_flags+="[no_contin]"
# test "$USE_MERGE_DRUMS" == true && corpus_path_flags+="[merge_drums]"

corpus_dir_path="data/corpus/${DATA_NAME}${corpus_path_flags}"
corpus_dir_path+="_tpq${TPQ}_r${MAX_TRACK_NUMBER}_d${MAX_DURATION}"
corpus_dir_path+="_v${VELOCITY_STEP}_t${TEMPO_MIN}_${TEMPO_MAX}_${TEMPO_STEP}"

do_midi_to_corpus=true
do_bpe=false
if [ -n "${BPE_ITER_NUM+x}" ] && [ "$BPE_ITER_NUM" -ne 0 ]; then
    do_bpe=true
    bpe_corpus_dir_path="${corpus_dir_path}"
    bpe_corpus_dir_path+="_bpe${BPE_ITER_NUM}_${ADJACENCY}_${SAMPLE_RATE}"

    if [ -d "$bpe_corpus_dir_path" ] \
        && {
            [ -f "${bpe_corpus_dir_path}/corpus" ] \
            || [ -f "${bpe_corpus_dir_path}/arrays.npz" ]; \
        } \
        && [ -f "${bpe_corpus_dir_path}/shape_vocab" ]; then
        
        echo "BPE Output directory: $bpe_corpus_dir_path already" \
            "has corpus and shape_vocab file." \
            | tee -a "$log_path"

        if [ -n "$use_existed_flag" ]; then
            printf "Flag --use-existed is set\nLearn bpe vocab is skipped\n" \
                | tee -a "$log_path"
            do_bpe=false
            do_midi_to_corpus=false

        else
            printf "Remove?\n(y=remove/n=skip bpe):" | tee -a "$log_path"
            read -r yn

            # this grammar (the #[] operator) means in the variable $yn,
            # any Y or y in first position will be dropped if they exist.
            if [ "$yn" == "${yn#[Yy]}" ]; then 
                # enter this block if yn != [Yy]
                do_bpe=false
                do_midi_to_corpus=false
                echo "Learn bpe vocab is skipped" | tee -a "$log_path"
            else
                rm -f "${bpe_corpus_dir_path}/*"
            fi
        fi
    fi
fi

if [ "$do_midi_to_corpus" == true ]; then
    midi_to_corpus_flags=()

    test "$CONTINUING_NOTE" == true && \
        midi_to_corpus_flags+=("--use-continuing-note")

    # test "$USE_MERGE_DRUMS" == true && \
    #     midi_to_corpus_flags+=("--use-merge-drums")

    test "$MIDI_TO_CORPUS_VERBOSE" == true && \
        midi_to_corpus_flags+=("--verbose")

    if [ "${#midi_to_corpus_flags[@]}" -ne 0 ]; then
        echo "Added '${midi_to_corpus_flags[*]}' to midi_to_corpus argument" \
            | tee -a "$log_path"
    fi

    echo "Corpus dir: ${corpus_dir_path}"

    python3 midi_to_corpus.py \
        --tpq "$TPQ" \
        --max-track-number "$MAX_TRACK_NUMBER" \
        --max-duration "$MAX_DURATION" \
        --velocity-step "$VELOCITY_STEP" \
        --tempo-quantization "$TEMPO_MIN" "$TEMPO_MAX" "$TEMPO_STEP" \
        --log "$log_path" --worker-number "$MIDI_WORKER_NUMBER" \
        --output-dir-path "$corpus_dir_path" \
        --recursive $use_existed_flag "${midi_to_corpus_flags[@]}" \
        -- "$MIDI_DIR_PATH" \
        || {
            echo "midi_to_corpus.py failed. pipeline.sh exit." \
                | tee -a "$log_path"
            exit 1;
        }
fi

if [ "$do_bpe" == true ]; then
    echo "Start learn bpe vocab" | tee -a "$log_path"
    
    # compile
    make -C ./bpe || {
        echo "BPE program compile error. pipeline.sh exit." \
            | tee -a "$log_path"
        exit 1;
    }

    # create new dir 
    if [ -d "$bpe_corpus_dir_path" ]; then
        rm -f "${bpe_corpus_dir_path}/*"
    else
        mkdir "$bpe_corpus_dir_path"
    fi

    # copy paras and pathlist
    cp "${corpus_dir_path}/paras" "$bpe_corpus_dir_path"
    cp "${corpus_dir_path}/pathlist" "$bpe_corpus_dir_path"

    # run learn_vocab
    bpe/learn_vocab \
        "$BPE_LOG" "$corpus_dir_path" "$bpe_corpus_dir_path" "$BPE_ITER_NUM" \
        "$ADJACENCY" "$SAMPLE_RATE" "$MIN_SCORE_LIMIT" "$BPE_WORKER_NUMBER" \
        | tee -a "$log_path"
    
    bpe_exit_code=${PIPESTATUS[0]}
    if [ "$bpe_exit_code" -ne 0 ]; then
        echo "learn_vocab failed. pipeline.sh exit." | tee -a "$log_path"
        echo "Args:"
        echo "$BPE_LOG $corpus_dir_path $bpe_corpus_dir_path $BPE_ITER_NUM" \
            "$ADJACENCY $SAMPLE_RATE $MIN_SCORE_LIMIT $BPE_WORKER_NUMBER"
        rm -r "$bpe_corpus_dir_path"
        exit 1;
    fi

    # process bpe log
    python3 plot_bpe_log.py "$bpe_corpus_dir_path" "$log_path"

    # check if tokenized corpus is equal to original corpus
    echo "Begin equality verification"
    python3 verify_corpus_equality.py \
        "$corpus_dir_path" "$bpe_corpus_dir_path" "$MIDI_WORKER_NUMBER" \
        || {
            echo "Equality verification failed. pipeline.sh exit." \
                | tee -a "$log_path";
            exit 1;
        }
    echo "Equality verification success"
fi

if [ -n "${BPE_ITER_NUM+x}" ] && [ "$BPE_ITER_NUM" -ne 0 ]; then
    # replace corpus_dir_path to bpe_corpus_dir_path
    corpus_dir_path="$bpe_corpus_dir_path"
    bpe_flag=("--bpe")
fi

python3 make_arrays.py \
    "${bpe_flag[@]}" --debug $use_existed_flag --log "$log_path" \
    --worker-number "$MIDI_WORKER_NUMBER" -- "$corpus_dir_path" \
    || {
        echo "text_to_array.py failed. pipeline.sh exit." \
            | tee -a "$log_path";
        exit 1;
    }

######## TRAIN MODEL ########

# test if NO_TRAIN is a set variables
if [ -n "${NO_TRAIN+x}" ]; then
    echo "No training. pipeline.sh exit." | tee -a "$log_path"
    exit 0
fi

model_dir_path="models/$(date '+%Y%m%d-%H%M%S')-"$full_config_name
if [ -d "$model_dir_path" ]; then
    rm -rf "$model_dir_path"
else
    mkdir "$model_dir_path"
    mkdir "${model_dir_path}/ckpt"
    mkdir "${model_dir_path}/eval_samples"
fi
echo "Model dir: $model_dir_path"

train_flags=()
test "$PERMUTE_MPS" == true          && train_flags+=("--permute-mps")
test "$NOT_USE_MPS_NUMBER" == true   && train_flags+=("--not-use-mps-number")
test "$PERMUTE_TRACK_NUMBER" == true && train_flags+=("--permute-track-number")
test "$USE_LINEAR_ATTENTION" == true && train_flags+=("--use-linear-attn")
test "$FLATTEN_VIRTUAL_PIECES" == true && train_flags+=("--flatten-virtual-pieces")

if [ "${#train_flags[@]}" -ne 0 ]; then
    echo "Added '${train_flags[*]}' to train.py's argument" \
        | tee -a "$log_path"
fi

launch_command="python3"
if [ "$USE_PARALLEL" == true ]; then
    NUM_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," " " | wc -w;)
    NUM_TOTAL_DEVICE=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_TOTAL_DEVICE" -lt "$NUM_DEVICES" ]; then
        NUM_DEVICES=$NUM_TOTAL_DEVICE
    fi
    if [ "$NUM_DEVICES" == "1" ]; then
        USE_PARALLEL=false
    else
        accelerate config default
        launch_command="accelerate launch --multi_gpu --num_machines 1"
        launch_command+=" --num_processes $NUM_DEVICES"
        train_flags+=("--use-parallel")
    fi
fi

$launch_command train.py \
    --test-paths-file "$TEST_PATHS_FILE" \
    --valid-paths-file "$VALID_PATHS_FILE" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --pitch-augmentation-range "$PITCH_AUGMENTATION_RANGE" \
    --virtual-piece-step-ratio "$VIRTUAL_PIECE_STEP_RATIO" \
    \
    --layers-number "$LAYERS_NUMBER" --attn-heads-number "$ATTN_HEADS_NUMBER" \
    --embedding-dim "$EMBEDDING_DIM" \
    --batch-size "$BATCH_SIZE" --max-updates "$MAX_UPDATES" \
    --validation-interval "$VALIDATION_INTERVAL" --early-stop "$EARLY_STOP" \
    --max-grad-norm "$MAX_GRAD_NORM" --loss-padding "$LOSS_PADDING" \
    \
    --lr-peak "$LEARNING_RATE_PEAK" \
    --lr-warmup-updates "$LEARNING_RATE_WARMUP_UPDATES" \
    --lr-decay-end-updates "$LEARNING_RATE_DECAY_END_UPDATES" \
    --lr-decay-end-ratio "$LEARNING_RATE_DECAY_END_RATIO" \
    \
    --softmax-temperature "$SOFTMAX_TEMPERATURE" \
    --sample-function "$SAMPLE_FUNCTION" \
    --sample-threshold "$SAMPLE_THRESHOLD" \
    --valid-eval-sample-number "$VALID_EVAL_SAMPLE_NUMBER" \
    --valid-eval-worker-number "$EVAL_WORKER_NUMBER" \
    \
    --max-pieces-per-gpu "$MAX_PIECE_PER_GPU" --use-device "$USE_DEVICE" \
    --seed "$SEED" --log "$log_path" "${train_flags[@]}" \
    -- "$MIDI_DIR_PATH" "$corpus_dir_path" "$model_dir_path" \
    || {
        echo "Training failed. pipeline.sh exit." | tee -a "$log_path"
        exit 1;
    }

######## EVALUATION ########

# test if NO_EVAL is a set variables
if [ -n "${NO_EVAL+x}" ]; then
    echo "No evaluation" | tee -a "$log_path"
    echo "pipeline.sh exit."
    exit 0
fi

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
        echo "Evaluation failed. pipeline.sh exit." | tee -a "$log_path"
        exit 1;
    }

echo "All done. pipeline.sh exit." | tee -a "$log_path"
