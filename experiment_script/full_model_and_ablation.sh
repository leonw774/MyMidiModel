#!/bin/bash

# ABLATIONS
if [ $# -eq 0 ]; then
    echo "need dataset name"
    exit 1
else 
    if [ $# -eq 1 ]; then
        dataset_name=$1
        do_ablation='no'
    else
        if [ $# -eq 2 ] || [ $# -eq 3 ]; then
            dataset_name=$1
            do_ablation=""
            do_full=""
            arg_array=( "$@" )
            
            if [[ " ${arg_array[*]} " =~ " --full " ]]; then
                do_full="true"
            fi
            if [[ " ${arg_array[*]} " =~ " --ablation " ]]; then
                do_ablation="true"
            fi
        else
            echo "invalid argument"
            exit 1
        fi
    fi
fi

model_setting="vanilla_small"
# model_setting="test_cpu"

if [ "$do_full" == 'true' ]; then
    # full model
    ./pipeline.sh "$dataset_name" ours_sample1.0 $model_setting --use-existed
fi
if [ "$do_ablation" == 'true' ]; then
    # no BPE
    ONLY_EVAL_UNCOND=true ./pipeline.sh "$dataset_name" no_bpe $model_setting --use-existed

    # not use MPS order as position number
    ONLY_EVAL_UNCOND=true ./pipeline.sh "$dataset_name" ours_sample1.0 "${model_setting}_no_mps" --use-existed

    # no continuative duration encoding
    ONLY_EVAL_UNCOND=true ./pipeline.sh "${dataset_name}_no_contin" ours_sample1.0 vanilla_small --use-existed
fi
