#!/bin/bash

# experiment on applying vocabulary from corpus A to corpus B
base_path="data/corpus/experiment_apply_learned_bpe_on_other"
lmd_orig_path="data/corpus/lmd_full_nth32_r40_d32_v16_t24_200_16"
lmd_bpe_path="data/corpus/lmd_full_nth32_r40_d32_v16_t24_200_16_bpe128_ours_1.0"
snd_orig_path="data/corpus/snd_nth32_r40_d32_v16_t24_200_16"
snd_bpe_path="data/corpus/snd_nth32_r40_d32_v16_t24_200_16_bpe128_ours_1.0"
rm -r $base_path
mkdir $base_path
mkdir "${base_path}/corpus"
mkdir "${base_path}/logs"

# make sure bpe programs are new
make -C ./bpe

## vocab source -- applied corpus

mkdir "${base_path}/corpus/lmd--snd_ours_1.0"
./bpe/apply_vocab -log \
    $snd_orig_path \
    "${base_path}/corpus/lmd--snd_ours_1.0/corpus" \
    "${lmd_bpe_path}/shape_vocab" | tee "${base_path}/logs/lmd--snd_ours_1.0.log" -a

sed -i 's/\r/\n/g ; s/\x1B\[2K//g' ${base_path}/logs/lmd--snd_ours_1.0.log
python3 plot_bpe_log.py ${base_path}/corpus/lmd--snd_ours_1.0 ${base_path}/logs/lmd--snd_ours_1.0.log

python3 make_arrays.py --bpe --log "${base_path}/logs/lmd--snd_ours_1.0.log" --debug "${base_path}/corpus/lmd--snd_ours_1.0"


mkdir "${base_path}/corpus/snd--lmd_ours_1.0"
./bpe/apply_vocab -log \
    $lmd_orig_path \
    "${base_path}/corpus/snd--lmd_ours_1.0/corpus" \
    "${snd_bpe_path}/shape_vocab" | tee "${base_path}/logs/snd--lmd_ours_1.0.log" -a

sed -i 's/\r/\n/g ; s/\x1B\[2K//g' ${base_path}/logs/snd--lmd_ours_1.0.log
python3 plot_bpe_log.py ${base_path}/corpus/snd--lmd_ours_1.0 ${base_path}/logs/snd--lmd_ours_1.0.log

python3 make_arrays.py --bpe --log "${base_path}/logs/snd--lmd_ours_1.0.log" --debug "${base_path}/corpus/snd--lmd_ours_1.0"

