source config.sh
python3 midi/midi_2_text.py --nth ${NTH} --max-track-num ${MAX_TRACK_NUM} --max-duration ${MAX_DURATION} --tempo-quantization ${TEMPO_MIN} ${TEMPO_MAX} ${TEMPO_STEP} \
    -r -w 8 -o data/preprocessed_midi/raw_corpus.txt midi/midis