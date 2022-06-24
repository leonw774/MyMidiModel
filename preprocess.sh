source config.sh
PARAS_SUFFIX=nth${NTH}_r${MAX_TRACK_NUM}_d${MAX_DURATION}_v${VELOCITY_STEP}_t${TEMPO_MIN}-${TEMPO_MAX}-${TEMPO_STEP}
python3 data/midi_2_text.py --nth ${NTH} --max-track-number ${MAX_TRACK_NUMBER} --max-duration ${MAX_DURATION} \
    --velocity-step ${VELOCITY_STEP} --tempo-quantization ${TEMPO_MIN} ${TEMPO_MAX} ${TEMPO_STEP} -r -w 8 \
    -o data/processed_midi/corpus_${PARAS_SUFFIX}.txt data/midi
python3 --max-sample-length ${MAX_SAMPLE_LENGTH} data/make_vocab.py data/processed_midi/corpus_${PARAS_SUFFIX}.txt data/data/${PARAS_SUFFIX}