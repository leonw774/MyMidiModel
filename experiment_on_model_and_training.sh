#!/bin/bash

# experiments on training model design
./pipeline.sh lmd_full_posevent no_bpe default_linear --use-existed
./pipeline.sh lmd_full_posevent no_bpe permute_data_linear --use-existed
./pipeline.sh lmd_full_posevent no_bpe permute_data+loss_linear --use-existed

# experiment on input data design
./pipeline.sh lmd_full_posattr no_bpe default_linear --use-existed
./pipeline.sh lmd_full_posattr no_bpe permute_data_linear --use-existed
./pipeline.sh lmd_full_posattr no_bpe permute_data+loss_linear --use-existed
