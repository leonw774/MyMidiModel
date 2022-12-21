#!/bin/bash

# basic config
./pipeline.sh lmd_full_posevent no_bpe permute_data_small_linear --use-existed
./pipeline.sh snd_posevent      no_bpe permute_data_small_linear --use-existed

# experiments on BPE usage
./pipeline.sh lmd_full_posevent ours_sample1.0 permute_data_small_linear --use-existed
./pipeline.sh snd_posevent      ours_sample1.0 permute_data_small_linear --use-existed

# experiments on set loss
./pipeline.sh lmd_full_posevent ours_sample1.0 permute_data+loss_small_linear --use-existed
./pipeline.sh snd_posevent      ours_sample1.0 permute_data+loss_small_linear --use-existed

# experiment on input data design (??)
# - posevent v.s. posattr
# - no_time_signature & no_tempo in input array

# experiment on data augmentation (permutation & pitch-shift) (??)
#

