#!/bin/bash
./pipeline.sh symphonynet_same_setting no_bpe no_train
./pipeline.sh symphonynet_same_setting ours_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_same_setting ours_sample0.5 no_train --use-existed
./pipeline.sh symphonynet_same_setting symphonynet_sample1.0 no_train --use-existed
./pipeline.sh symphonynet_same_setting wordpiece_sample1.0 no_train --use-existed
