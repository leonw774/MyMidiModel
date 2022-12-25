#!/bin/bash

# basic config
./pipeline.sh lmd_full_posevent no_bpe linear_small_permute_data --use-existed
./pipeline.sh snd_posevent      no_bpe linear_small_permute_data --use-existed

# experiments on BPE usage
./pipeline.sh lmd_full_posevent ours_sample1.0 linear_small_permute_data --use-existed
./pipeline.sh snd_posevent      ours_sample1.0 linear_small_permute_data --use-existed

# experiments on set loss
./pipeline.sh lmd_full_posevent ours_sample1.0 linear_small_permute_data+loss --use-existed
./pipeline.sh snd_posevent      ours_sample1.0 linear_small_permute_data+loss --use-existed

# experiment on input data design (??)
# - posevent v.s. posattr
# - no_time_signature & no_tempo in input array

# experiment on data augmentation (permutation & pitch-shift) (??)
#

