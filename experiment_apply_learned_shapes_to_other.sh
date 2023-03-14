#!/bin/bash

# experiment on applying vocabulary from corpus A to corpus B
BASE_PATH="data/corpus/experiment_apply_learned_bpe_on_other"
LMD_ORIG_PATH="data/corpus/lmd_full_nth32_r32_d32_v16_t24_200_16"
LMD_BPE_PATH="data/corpus/lmd_full_nth32_r32_d32_v16_t24_200_16_bpe128_ours_1.0"
SND_ORIG_PATH="data/corpus/snd_nth32_r32_d32_v16_t24_200_16"
SND_BPE_PATH="data/corpus/snd_nth32_r32_d32_v16_t24_200_16_bpe128_ours_1.0"
rm -r $BASE_PATH
mkdir $BASE_PATH
mkdir "${BASE_PATH}/corpus"
mkdir "${BASE_PATH}/logs"

# make sure bpe programs are new
make -C ./bpe

## vocab source -- applied corpus

mkdir "${BASE_PATH}/corpus/lmd--snd_ours_1.0"
./bpe/apply_vocab -log \
    $SND_ORIG_PATH \
    "${BASE_PATH}/corpus/lmd--snd_ours_1.0/corpus" \
    "${LMD_BPE_PATH}/shape_vocab" | tee "${BASE_PATH}/logs/lmd--snd_ours_1.0.log" -a

sed -i 's/\r/\n/g ; s/\x1B\[2K//g' ${BASE_PATH}/logs/lmd--snd_ours_1.0.log
python3 plot_bpe_log.py ${BASE_PATH}/corpus/lmd--snd_ours_1.0 ${BASE_PATH}/logs/lmd--snd_ours_1.0.log

python3 make_arrays.py --bpe --log "${BASE_PATH}/logs/lmd--snd_ours_1.0.log" --debug "${BASE_PATH}/corpus/lmd--snd_ours_1.0"


mkdir "${BASE_PATH}/corpus/snd--lmd_ours_1.0"
./bpe/apply_vocab -log \
    $LMD_ORIG_PATH \
    "${BASE_PATH}/corpus/snd--lmd_ours_1.0/corpus" \
    "${SND_BPE_PATH}/shape_vocab" | tee "${BASE_PATH}/logs/snd--lmd_ours_1.0.log" -a

sed -i 's/\r/\n/g ; s/\x1B\[2K//g' ${BASE_PATH}/logs/snd--lmd_ours_1.0.log
python3 plot_bpe_log.py ${BASE_PATH}/corpus/snd--lmd_ours_1.0 ${BASE_PATH}/logs/snd--lmd_ours_1.0.log

python3 make_arrays.py --bpe --log "${BASE_PATH}/logs/snd--lmd_ours_1.0.log" --debug "${BASE_PATH}/corpus/snd--lmd_ours_1.0"
