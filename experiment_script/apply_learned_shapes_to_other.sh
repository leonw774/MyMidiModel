#!/bin/bash

# experiment on applying vocabulary from corpus A to corpus B
base_path="data/corpus/experiment_apply_learned_bpe_on_other"
lmd_orig_path="data/corpus/lmd_full_tpq12_r40_d48_v16_t8_240_8"
lmd_bpe_path="data/corpus/lmd_full_tpq12_r40_d48_v16_t8_240_8_bpe128_ours_1.0"
snd_orig_path="data/corpus/snd_tpq12_r40_d48_v16_t8_240_8"
snd_bpe_path="data/corpus/snd_tpq12_r40_d48_v16_t8_240_8_bpe128_ours_1.0"
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

python3 plot_bpe_log.py "${base_path}/corpus/lmd--snd_ours_1.0" "${base_path}/logs/lmd--snd_ours_1.0.log"

cp "${lmd_bpe_path}/shape_vocab" "${base_path}/corpus/lmd--snd_ours_1.0"
cp "${lmd_bpe_path}/paras" "${base_path}/corpus/lmd--snd_ours_1.0"
python3 make_arrays.py --bpe --log "${base_path}/logs/lmd--snd_ours_1.0.log" \
    --debug "${base_path}/corpus/lmd--snd_ours_1.0"


mkdir "${base_path}/corpus/snd--lmd_ours_1.0"
./bpe/apply_vocab -log \
    $lmd_orig_path \
    "${base_path}/corpus/snd--lmd_ours_1.0/corpus" \
    "${snd_bpe_path}/shape_vocab" | tee "${base_path}/logs/snd--lmd_ours_1.0.log" -a

python3 plot_bpe_log.py "${base_path}/corpus/snd--lmd_ours_1.0" "${base_path}/logs/snd--lmd_ours_1.0.log"

cp "${snd_bpe_path}/shape_vocab" "${base_path}/corpus/snd--lmd_ours_1.0"
cp "${snd_bpe_path}/paras" "${base_path}/corpus/snd--lmd_ours_1.0"
python3 make_arrays.py --bpe --log "${base_path}/logs/snd--lmd_ours_1.0.log" \
    --debug "${base_path}/corpus/snd--lmd_ours_1.0"

