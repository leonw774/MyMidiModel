#!/bin/bash
./make_data.sh symphonynet_same_setting test_bpe test_model
./make_data.sh symphonynet_same_setting ours_sample1.0 test_model
./make_data.sh symphonynet_same_setting symphonynet_sample1.0 test_model
