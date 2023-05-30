#!/bin/bash

# ABLATIONS
if [ $# -eq 0 ] ; then
    echo "need dataset name"
    exit 0
else 
    if [ $# -eq 1 ] ; then
        dataset_name=$1
    else
        if [ $# -eq 2 ] ; then
            dataset_name=$1
            do_ablation=$2
            if [ $do_ablation -ne '--ablation' ] then
                echo "invalid argument"
                exit 0
            fi
        fi
    fi
fi


# full component
./pipeline.sh $dataset_name ours_sample1.0 linear_base --use-existed
# ./pipeline.sh lmd_full ours_sample1.0 linear_base --use-existed

test -z "$do_ablation" && exit 0

# no BPE
./pipeline.sh $dataset_name no_bpe linear_base --use-existed
# ./pipeline.sh lmd_full no_bpe linear_base --use-existed

# not use MPS order as position number
./pipeline.sh $dataset_name ours_sample1.0 linear_base_no_mps --use-existed
# ./pipeline.sh lmd_full ours_sample1.0 linear_bas_no_MPS  --use-existed

# no continuative duration encoding
./pipeline.sh "${dataset_name}_no_contin" ours_sample1.0 linear_base --use-existed
# ./pipeline.sh lmd_full_no_contin ours_sample1.0 linear_base  --use-existed
