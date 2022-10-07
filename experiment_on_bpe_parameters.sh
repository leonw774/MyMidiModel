#!/bin/bash

# make original corpus first
./pipeline.sh lmd_full_posevent no_bpe no_train --use-existed
./pipeline.sh symphonynet_posevent no_bpe no_train --use-existed

# experiment on merge condition
./pipeline.sh lmd_full_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh lmd_full_posevent musicbpe_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent musicbpe_sample1.0 no_train --use-existed

# experiment on sample rate
./pipeline.sh lmd_full_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh lmd_full_posevent ours_sample0.1 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_sample0.1 no_train --use-existed

# experiment on scoring method on
./pipeline.sh lmd_full_posevent ours_wplike_sample1.0 no_train --use-existed
./pipeline.sh lmd_full_posevent ours_wplike_sample0.1 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_wplike_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_wplike_sample0.1 no_train --use-existed

# experiment on applying vocabulary from corpus A to corpus B
mkdir experiment_bpe_apply_result

./bpe/apply_vocab data/corpus/symphonynet_dataset_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_default_ours_1.0 \
    experiment_bpe_apply_result/lmd--symphonynet_default_ours_1.0 \
    data/corpus/lmd_full_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_default_ours_1.0/shape_vocab | tee experiment_bpe_apply_result/log_lmd--symphonynet_default_ours_1.0 -a

./bpe/apply_vocab data/corpus/symphonynet_dataset_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_wplike_ours_1.0 \
    experiment_bpe_apply_result/lmd--symphonynet_wplike_ours_1.0 \
    data/corpus/lmd_full_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_wplike_ours_1.0/shape_vocab | tee experiment_bpe_apply_result/log_lmd--symphonynet_wplike_ours_1.0 -a

./bpe/apply_vocab data/corpus/lmd_full_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_default_ours_1.0 \
    experiment_bpe_apply_result/symphonynet--lmd_default_ours_1.0 \
    data/corpus/symphonynet_dataset_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_default_ours_1.0/shape_vocab | tee experiment_bpe_apply_result/log_symphonynet--lmd_default_ours_1.0 -a

./bpe/apply_vocab data/corpus/lmd_full_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_wplike_ours_1.0 \
    experiment_bpe_apply_result/symphonynet--lmd_wplike_ours_1.0 \
    data/corpus/symphonynet_dataset_nth96_r32_d96_v16_t24_200_16_posevent_bpe1000_wplike_ours_1.0/shape_vocab | tee experiment_bpe_apply_result/log_lmd--symphonynet--lmd_wplike_ours_1.0 -a
