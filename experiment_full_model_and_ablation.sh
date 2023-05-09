#!/bin/bash

# full component
./pipeline.sh snd      ours_sample1.0 linear_base --use-existed
./pipeline.sh lmd_full ours_sample1.0 linear_base --use-existed

# ABLATIONS
if [ $# -ne 1 ] ; then
    exit 0
else
    if [ $1 -ne '--ablation' ] then
        exit 0
    fi
fi

# no BPE
./pipeline.sh snd      no_bpe linear_base --use-existed
./pipeline.sh lmd_full no_bpe linear_base --use-existed

# no context encoding
./pipeline.sh snd      ours_sample1.0 linear_base_no_context --use-existed
./pipeline.sh lmd_full ours_sample1.0 linear_base_no_context --use-existed

# no auxiliary task of instrument class prediction
./pipeline.sh snd      ours_sample1.0 linear_base_no_output_instr --use-existed
./pipeline.sh lmd_full ours_sample1.0 linear_bas_no_output_instr  --use-existed

# no continuative duration encoding
./pipeline.sh snd_no_contin      ours_sample1.0 linear_base --use-existed
./pipeline.sh lmd_full_no_contin ours_sample1.0 linear_base  --use-existed
