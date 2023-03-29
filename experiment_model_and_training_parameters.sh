#!/bin/bash

# full component
./pipeline.sh lmd_full ours_sample1.0 linear_base --use-existed
./pipeline.sh snd      ours_sample1.0 linear_base --use-existed

# ABLATIONS
if [ $# -ne 1 ] ; then
    exit 0
else
    if [ $1 -ne 'ablation' ] then
        exit 0
    fi
fi

# no BPE
./pipeline.sh lmd_full no_bpe linear_base --use-existed
./pipeline.sh snd      no_bpe linear_base --use-existed

# no data augmentation (no permutations)
# ./pipeline.sh lmd_full ours_sample1.0 linear_base_no_data_aug --use-existed
# ./pipeline.sh snd      ours_sample1.0 linear_base_no_data_aug --use-existed

