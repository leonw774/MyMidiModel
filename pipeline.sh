#!/bin/bash
echo "pipeline.sh start."
use_existed=""
if [ $# -eq 4 ]; then
    if [ $4 == '--use-existed' ]; then
        use_existed="--use-existed"
    else
        echo "Expect arguments to be three configuration file name for midi preprocessing, bpe, and training/model setting, and an optional '--use-existed' flag at the fourth position."
        exit 1
    fi
else
    if [ $# -ne 3 ]; then
        echo "Expect arguments to be three configuration file name for midi preprocessing, bpe, and training/model setting, and an optional '--use-existed' flag at the fourth position."
        exit 1
    fi
fi

# check if all argument is a file and execute them to get their vars
full_config_name=$1"-"$2"-"$3
corpus_config_file_path="configs/corpus/"$1".sh"
bpe_config_file_path="configs/bpe/"$2".sh"
train_config_file_path="configs/train/"$3".sh"

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

log_path="logs/$(date '+%Y%m%d-%H%M%S')-${full_config_name}.log"
echo "Log file: $log_path"
touch $log_path

######## MAKE CORPUS ########

corpus_dir_path="data/corpus/${DATA_NAME}_nth${NTH}_r${MAX_TRACK_NUMBER}_d${MAX_DURATION}_v${VELOCITY_STEP}_t${TEMPO_MIN}_${TEMPO_MAX}_${TEMPO_STEP}"

do_midi_to_corpus=true
do_bpe=false
if [ $BPE_ITER -ne 0 ]; then
    do_bpe=true
    bpe_corpus_dir_path="${corpus_dir_path}_bpe${BPE_ITER}_${MERGE_CONDITION}_${SAMPLE_RATE}"
    if [ -d $bpe_corpus_dir_path ] && [ -f "${bpe_corpus_dir_path}/corpus" ] && [ -f "${bpe_corpus_dir_path}/shape_vocab" ]; then
        if [ -n "$use_existed" ]; then
            echo "BPE Output directory: $bpe_corpus_dir_path already has corpus and shape_vocab file." | tee -a $log_path
            echo "Flag --use-existed is set" | tee -a $log_path
            echo "Learn bpe vocab is skipped" | tee -a $log_path
            do_bpe=false
            do_midi_to_corpus=false
        else
            echo "BPE Output directory: $bpe_corpus_dir_path already has corpus and shape_vocab file. Remove? (y=remove/n=skip bpe)" | tee -a $log_path
            read yn
            if [ "$yn" == "${yn#[Yy]}" ]; then 
            # this grammar (the #[] operator) means that, in the variable $yn, any Y or y in 1st position will be dropped if they exist.
                # enter this block if yn != [Yy]
                do_bpe=false
                do_midi_to_corpus=false
                echo "Learn bpe vocab is skipped" | tee -a $log_path
            else
                rm -f "${bpe_corpus_dir_path}/*"
            fi
        fi
    fi
fi

if [ "$do_midi_to_corpus" == true ]; then
    midi_to_corpus_other_args=""
    test "$CONTINUING_NOTE" == true && midi_to_corpus_other_args="$midi_to_corpus_other_args --use-continuing-note"
    test "$USE_MERGE_DRUMS" == true && midi_to_corpus_other_args="$midi_to_corpus_other_args --use-merge-drums"
    test "$MIDI_TO_CORPUS_VERBOSE" == true && midi_to_corpus_other_args="$midi_to_corpus_other_args --verbose"
    test -n "$train_other_args" && { echo "Appended $midi_to_corpus_other_args to midi_to_corpus's argument" | tee -a $log_path ; }

    echo "Corpus dir: ${corpus_dir_path}"

    python3 midi_to_corpus.py --nth $NTH --max-track-number $MAX_TRACK_NUMBER --max-duration $MAX_DURATION --velocity-step $VELOCITY_STEP \
        --tempo-quantization $TEMPO_MIN $TEMPO_MAX $TEMPO_STEP --position-method $POSITION_METHOD $midi_to_corpus_other_args $use_existed \
        --log $log_path -w $PROCESS_WORKERS -r -o $corpus_dir_path $MIDI_DIR_PATH
    test $? -ne 0 && { echo "midi_to_text.py failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1
fi

if [ "$do_bpe" == true ]; then
    echo "Start learn bpe vocab" | tee -a $log_path
    # compile
    make -C ./bpe
    test $? -ne 0 && { echo "Compile error. pipeline.sh exit." | tee -a $log_path ; } && exit 1

    # create new dir 
    if [ -d $bpe_corpus_dir_path ]; then
        rm -f "${bpe_corpus_dir_path}/*"
    else
        mkdir $bpe_corpus_dir_path
    fi

    # copy paras and pathlist
    cp "${corpus_dir_path}/paras" $bpe_corpus_dir_path
    cp "${corpus_dir_path}/pathlist" $bpe_corpus_dir_path

    # run learn_vocab
    bpe/learn_vocab $BPE_DOLOG $BPE_CLEARLINE $corpus_dir_path $bpe_corpus_dir_path $BPE_ITER $MERGE_CONDITION $SAMPLE_RATE $MIN_SCORE_LIMIT $BPE_WORKER | tee -a $log_path
    
    bpe_exit_code=${PIPESTATUS[0]}
    if [ $bpe_exit_code -ne 0 ]; then
        echo "learn_vocab failed. exit code: $bpe_exit_code. pipeline.sh exit." | tee -a $log_path
        echo "bpe/learn_vocab $BPE_DOLOG $BPE_CLEARLINE $corpus_dir_path $bpe_corpus_dir_path $BPE_ITER $MERGE_CONDITION $SAMPLE_RATE $MIN_SCORE_LIMIT $BPE_WORKER"
        rm -r "$bpe_corpus_dir_path"
        exit 1
    fi

    # process bpe log
    echo "sed -i 's/\r/\n/g ; s/\x1B\[2K//g' $log_path"
    sed -i 's/\r/\n/g ; s/\x1B\[2K//g' $log_path
    python3 plot_bpe_log.py $bpe_corpus_dir_path $log_path

    # check if tokenized corpus is equal to original corpus
    python3 verify_corpus_equality.py $corpus_dir_path $bpe_corpus_dir_path $PROCESS_WORKERS | tee -a $log_path
    verify_exit_code=${PIPESTATUS[0]}
    test $verify_exit_code -ne 0 && { echo "Corpus equality verification failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1
fi

if [ $BPE_ITER -ne 0 ]; then
    # replace corpus_dir_path to bpe_corpus_dir_path
    corpus_dir_path=$bpe_corpus_dir_path
    BPE_OPTION="--bpe"
fi

python3 make_arrays.py --debug $BPE_OPTION --mp-worker-number $PROCESS_WORKERS --log $log_path $corpus_dir_path $use_existed
test $? -ne 0 && { echo "text_to_array.py failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

######## TRAIN MODEL ########

# test if NO_TRAIN is a set variables
if [ -n "${NO_TRAIN+x}" ]; then
    echo "No training" | tee -a $log_path
    echo "pipeline.sh exit."
    exit 0
fi

model_dir_path="models/$(date '+%Y%m%d-%H%M%S')-"$full_config_name
if [ -d $model_dir_path ]; then
        rm -rf $model_dir_path
    else
        mkdir $model_dir_path
        mkdir "${model_dir_path}/ckpt"
        mkdir "${model_dir_path}/eval_samples"
    fi
echo "Model dir: $model_dir_path"

train_other_args=""
if [ "$USE_PERMUTABLE_SUBSEQ_LOSS" == true ]; then
    python3 -c "import torch;import mps_loss" 2>&1 | grep ModuleNotFoundError
    # if module not exist, install
    if [ $? -eq 0 ]; then
        cd util/pytorch/mps_loss
        python3 setup.py install
        cd ../../..
        # check if success
        python3 -c "import torch;import mps_loss" 2>&1 | grep ModuleNotFoundError
        test $? -eq 0 && { echo "mps_loss extension setup failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1
    fi
    train_other_args="$train_other_args --use-permutable-subseq-loss"
fi
test "$PERMUTE_MPS" == true            && train_other_args="$train_other_args --permute-mps"
test "$PERMUTE_TRACK_NUMBER" == true   && train_other_args="$train_other_args --permute-track-number"
test "$USE_LINEAR_ATTENTION" == true   && train_other_args="$train_other_args --use-linear-attn"
test "$INPUT_CONTEXT" == true          && train_other_args="$train_other_args --input-context"
test "$INPUT_INSTRUMENTS" == true      && train_other_args="$train_other_args --input-instruments"
test "$OUTPUT_INSTRUMENTS" == true     && train_other_args="$train_other_args --output-instruments"
test -n "$MAX_PIECE_PER_GPU"           && train_other_args="$train_other_args --max-pieces-per-gpu $MAX_PIECE_PER_GPU"
test -n "$SEED"                        && train_other_args="$train_other_args --seed $SEED"
test -n "$train_other_args" && { echo "Appended $train_other_args to train.py's argument" | tee -a $log_path ; }

if [ "$USE_PARALLEL" == true ]; then
    num_CUDA_VISIBLE_DEVICE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c ;)
    num_CUDA_VISIBLE_DEVICE=$(($num_CUDA_VISIBLE_DEVICE+1)) # arithmetic expression
    num_CUDA_DEVICE=$(nvidia-smi --list-gpus | wc -l)
    if [ $num_CUDA_DEVICE -lt $num_CUDA_VISIBLE_DEVICE ]; then
        num_CUDA_VISIBLE_DEVICE=$num_CUDA_DEVICE
    fi
    if [ $num_CUDA_VISIBLE_DEVICE == "1" ]; then
        launch_command="python3"
        $USE_PARALLEL == false
    else
        accelerate config default
        launch_command="accelerate launch --multi_gpu --num_processes $num_CUDA_VISIBLE_DEVICE --num_machines 1"
        train_other_args="$train_other_args --use-parallel"
    fi
else
    launch_command="python3"
fi
$launch_command train.py \
    --max-seq-length $MAX_SEQ_LENGTH --pitch-augmentation-range $PITCH_AUGMENTATION_RANGE --measure-sample-step-ratio $MEASURE_SAMPLE_STEP_RATIO \
    --layers-number $LAYERS_NUMBER --attn-heads-number $ATTN_HEADS_NUMBER --embedding-dim $EMBEDDING_DIM \
    --batch-size $BATCH_SIZE --max-updates $MAX_UPDATES --grad-clip-norm $GRAD_CLIP_NORM --split-ratio $SPLIT_RATIO \
    --validation-interval $VALIDATION_INTERVAL --early-stop $EARLY_STOP \
    --generate-interval $GENERATE_INTERVAL --nucleus-sampling-threshold $NUCLEUS_THRESHOLD \
    --lr-peak $LEARNING_RATE_PEAK --lr-warmup-updates $LEARNING_RATE_WARMUP_UPDATES \
    --lr-decay-end-updates $LEARNING_RATE_DECAY_END_UPDATES --lr-decay-end-ratio $LEARNING_RATE_DECAY_END_RATIO \
    --use-device $USE_DEVICE --primer-measure-length $PRIMER_LENGTH --log $log_path $train_other_args $corpus_dir_path $model_dir_path

test $? -ne 0 && { echo "training failed. pipeline.sh exit." | tee -a $log_path ; } && exit 1

######## EVALUATION ########

# test if NO_EVAL is a set variables
if [ -n "${NO_EVAL+x}" ]; then
    echo "No evaluation" | tee -a $log_path
    echo "pipeline.sh exit."
    exit 0
fi

./evaluated_model.sh $MIDI_DIR_PATH $EVAL_SAMPLE_NUMBER $PROCESS_WORKERS $PRIMER_LENGTH $log_path $model_dir_path $NUCLEUS_THRESHOLD $SEED
test $? -eq 0 && echo "All done. pipeline.sh exit." | tee -a $log_path
