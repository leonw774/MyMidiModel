#!/bin/bash

# make original corpus first
./pipeline.sh lmd_full_posevent no_bpe no_train --use-existed
./pipeline.sh snd_posevent no_bpe no_train --use-existed

# experiment on merge condition
./pipeline.sh lmd_full_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh lmd_full_posevent mulpi_sample1.0 no_train --use-existed
./pipeline.sh snd_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh snd_posevent mulpi_sample1.0 no_train --use-existed

# experiment on sample rate
# ./pipeline.sh lmd_full_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh lmd_full_posevent ours_sample0.1 no_train --use-existed
# ./pipeline.sh snd_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh snd_posevent ours_sample0.1 no_train --use-existed

# experiment on scoring method
./pipeline.sh lmd_full_posevent ours_wplike_sample1.0 no_train --use-existed
./pipeline.sh lmd_full_posevent ours_wplike_sample0.1 no_train --use-existed
./pipeline.sh snd_posevent ours_wplike_sample1.0 no_train --use-existed
./pipeline.sh snd_posevent ours_wplike_sample0.1 no_train --use-existed
