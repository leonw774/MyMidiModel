#!/bin/bash

# full component
./pipeline.sh lmd_full ours_sample1.0 linear_large_permute_data+mps_loss --use-existed
./pipeline.sh snd      ours_sample1.0 linear_large_permute_data+mps_loss --use-existed

# ABLATIONS
if [ $# -ne 1 ] ; then
    exit 0
else
    if [ $1 -ne 'ablation' ] then
        exit 0
    fi
fi

# no BPE
./pipeline.sh lmd_full no_bpe linear_large_permute_data+mps_loss --use-existed
./pipeline.sh snd      no_bpe linear_large_permute_data+mps_loss --use-existed

# no MPS loss
./pipeline.sh lmd_full ours_sample1.0 linear_large_permute_data --use-existed
./pipeline.sh snd      ours_sample1.0 linear_large_permute_data --use-existed

# no BPE & no MPS loss
./pipeline.sh lmd_full no_bpe linear_large_permute_data --use-existed
./pipeline.sh snd      no_bpe linear_large_permute_data --use-existed

# no data augmentation (permutation & pitch-shift) & no MPS loss
# ./pipeline.sh lmd_full ours_sample1.0 linear_large --use-existed
# ./pipeline.sh snd      ours_sample1.0 linear_large --use-existed

