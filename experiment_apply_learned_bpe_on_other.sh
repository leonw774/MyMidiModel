#!/bin/bash

# experiment on applying vocabulary from corpus A to corpus B
BASE_PATH="data/corpus/experiment_apply_learned_bpe_on_other"
LMD_ORIG_PATH="data/corpus/lmd_full_nth96_r32_d96_v16_t24_200_16_posevent"
LMD_DEFAULT_PATH="data/corpus/lmd_full_nth96_r32_d96_v16_t24_200_16_posevent_bpe128_default_ours_1.0"
SND_ORIG_PATH="data/corpus/snd_nth96_r32_d96_v16_t24_200_16_posevent"
SND_DEFAULT_PATH="data/corpus/snd_nth96_r32_d96_v16_t24_200_16_posevent_bpe128_default_ours_1.0"
mkdir ${BASE_PATH}
mkdir "${BASE_PATH}/corpus"
mkdir "${BASE_PATH}/logs"

## vocab source -- applied corpus

mkdir "${BASE_PATH}/corpus/lmd--snd_default_ours_1.0"
./bpe/apply_vocab \
    $SND_ORIG_PATH \
    "${BASE_PATH}/corpus/lmd--snd_default_ours_1.0/corpus" \
    "${LMD_DEFAULT_PATH}/shape_vocab" | tee "${BASE_PATH}/logs/lmd--snd_default_ours_1.0.log" -a

python3 text_to_array.py --bpe 128 --log "${BASE_PATH}/logs/lmd--snd_default_ours_1.0.log" --debug "${BASE_PATH}/corpus/lmd--snd_default_ours_1.0"


mkdir "${BASE_PATH}/corpus/snd--lmd_default_ours_1.0"
./bpe/apply_vocab \
    $LMD_ORIG_PATH \
    "${BASE_PATH}/corpus/snd--lmd_default_ours_1.0/corpus" \
    "${SND_DEFAULT_PATH}/shape_vocab" | tee "${BASE_PATH}/logs/snd--lmd_default_ours_1.0.log" -a

python3 text_to_array.py --bpe 128 --log "${BASE_PATH}/logs/snd--lmd_default_ours_1.0.log" --debug "${BASE_PATH}/corpus/snd--lmd_default_ours_1.0"

