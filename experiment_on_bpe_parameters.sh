#!/bin/bash

# make original corpus first
./pipeline.sh lmd_full no_bpe no_train --use-existed
./pipeline.sh snd      no_bpe no_train --use-existed

# make multi-note bpe compressed corpus
./pipeline.sh lmd_full ours_sample1.0 no_train --use-existed
./pipeline.sh snd      ours_sample1.0 no_train --use-existed

# use mulpi as merge condition (to see if same as music bpe)
./pipeline.sh lmd_full mulpi_sample1.0 no_train --use-existed
./pipeline.sh snd      mulpi_sample1.0 no_train --use-existed

# experiment on sample rate
./pipeline.sh lmd_full ours_sample0.1 no_train --use-existed
./pipeline.sh snd      ours_sample0.1 no_train --use-existed
