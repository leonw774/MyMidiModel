#!/bin/bash

# experiment on merge condition
./pipeline.sh lmd_posevent ours_sample1.0 no_train
./pipeline.sh symphonynet_posevent ours_sample1.0 no_train
./pipeline.sh lmd_posevent musicbpe_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent musicbpe_sample1.0 no_train --use-existed

# experiment on sample rate
./pipeline.sh lmd_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_sample1.0 no_train --use-existed
./pipeline.sh lmd_posevent ours_sample0.1 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_sample0.1 no_train --use-existed

# experiment on scoring method on
./pipeline.sh lmd_posevent ours_wplike_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_wplike_sample1.0 no_train --use-existed
./pipeline.sh lmd_posevent ours_wplike_sample0.1 no_train --use-existed
./pipeline.sh symphonynet_posevent ours_wplike_sample0.1 no_train --use-existed
