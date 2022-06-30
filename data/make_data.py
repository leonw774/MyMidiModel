import json
import numpy as np
import os
import random
from argparse import ArgumentParser
from time import time

from midi_util import file_to_paras_and_pieces
from tokens import *


def piece_to_numpy(piece: str, vocabs: dict) -> np.ndarray:
    text_list = piece.split(' ')
    x = np.full((len(text_list), 7), fill_value=-1, dtype=np.int16)
    # event, duration, velocity, track_numer, instrument, position, measure_number

    dur_pad_id = vocabs['duration_tokens']['text2id']['[PAD]']
    vel_pad_id = vocabs['velocity_tokens']['text2id']['[PAD]']
    trn_pad_id = vocabs['track_number_tokens']['text2id']['[PAD]']
    ins_pad_id = vocabs['duration_tokens']['text2id']['[PAD]']

    cur_position = 0
    # head section and eos is measure 0
    # measure index begin at 1
    cur_measure_number = 0
    for i, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue
        if typename == 'B' or typename == 'E':
            x[i] = [
                vocabs['event_tokens']['text2id'][text],
                dur_pad_id,
                vel_pad_id,
                trn_pad_id,
                ins_pad_id,
                0,
                0
            ]
        elif typename == 'M':
            cur_position = 0
            cur_measure_number += 1
            x[i] = [
                vocabs['event_tokens']['text2id'][text],
                dur_pad_id,
                vel_pad_id,
                trn_pad_id,
                ins_pad_id,
                cur_position,
                cur_measure_number
            ]
        elif typename == 'R':
            track_number, instrument = text[1:].split(':')
            x[i] = [
                vocabs['event_tokens']['text2id']['R'],
                dur_pad_id,
                vel_pad_id,
                tokenstr2int(track_number),
                tokenstr2int(instrument),
                0,
                cur_measure_number
            ]
        else:
            event_text, *attr = text.split(':')
            if typename == 'P':
                cur_position = tokenstr2int(text[1:])
                x[i] = [
                    vocabs['event_tokens']['text2id'][event_text],
                    dur_pad_id,
                    vel_pad_id,
                    trn_pad_id,
                    ins_pad_id,
                    cur_position,
                    cur_measure_number
                ]
            elif typename == 'N':
                x[i] = [
                    vocabs['event_tokens']['text2id'][event_text],
                    vocabs['duration_tokens']['text2id'][attr[0]],
                    vocabs['velocity_tokens']['text2id'][attr[1]],
                    vocabs['track_number_tokens']['text2id'][attr[2]],
                    vocabs['instrument_tokens']['text2id'][tokenint2str(x[tokenstr2int(attr[2])+1, 4])],
                    cur_position,
                    cur_measure_number
                ]
            elif typename == 'T':
                x[i] = [
                    vocabs['event_tokens']['text2id'][event_text],
                    dur_pad_id,
                    vel_pad_id,
                    trn_pad_id,
                    ins_pad_id,
                    0, 
                    cur_measure_number
                ]
            else:
                x[i] = [
                    vocabs['event_tokens']['text2id']['UNK'],
                    dur_pad_id,
                    vel_pad_id,
                    trn_pad_id,
                    ins_pad_id,
                    0,
                    0
                ]
    return x


def make_token_dict(token_list: list) -> dict:
    token_dict = {
        'id2text': {i:t for i, t in enumerate(token_list)},
        'text2id': {t:i for i, t in enumerate(token_list)},
        'size': len(token_list),
    }
    return token_dict


def build_vocabs(pieces: list, paras: dict, max_sample_length: str):
    start_time = time()
    # Event tokens include:
    #   note pitches, positions in measure, end-of-position, measure numbers, time signatures,
    #   tempos, track, begin-of-score and end-of-score
    # pitches, positions and tempos are determined by arguments
    # although some positions may never appear in the corpus, which could be discarded
    # time signature is sampled from corpus
    # measure number size = max amount of measures that a max_sample_length can fit in
    pitche_tokens = [
        'N'+tokenint2str(i) for i in range(128)
    ]
    position_tokens = [
        'P'+tokenint2str(i) for i in range(get_theoretical_max_position(paras['nth']))
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    tempo_tokens = [
        'T'+tokenint2str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]
    track_tokens = ['R']

    # max measure number in all pieces
    max_measure_length = 0
    # the most number of measures span that can fit in the size of max_sample_length
    max_fitting_measure_number = 0
    measure_tokens = set()
    corpus_position_tokens = set()
    corpus_duration_tokens = set()
    for piece in pieces:

        text_list = piece.split(' ')
        measure_indices = [i for i, t in enumerate(text_list) if t == 'M']
        max_measure_length = max(max_measure_length, len(measure_indices))
        cur_fitting_measure_number = 0
        for size in range(len(measure_indices)-1, 1, -1):
            for start in range(0, len(measure_indices)-size):
                if measure_indices[start+size] - measure_indices[start] < max_sample_length:
                    cur_fitting_measure_number = size
                    break
            if cur_fitting_measure_number > 0:
                break
        max_fitting_measure_number = max(cur_fitting_measure_number, max_fitting_measure_number)

        measures = {t for t in text_list if t[0] == 'M'}
        positions = {t for t in text_list if t[0] == 'P'}
        durations = {t.split(':')[1] for t in text_list if t[0] == 'N'}
        measure_tokens.update(measures)
        corpus_position_tokens.update(positions)
        corpus_duration_tokens.update(durations)
    measure_tokens = list(measure_tokens)
    corpus_position_tokens = list(corpus_position_tokens)
    corpus_duration_tokens = list(corpus_duration_tokens)

    # place special token in front so that they have same index across all vocabulary
    event_tokens = SPECIAL_TOKENS + ['BOS', 'EOS'] + track_tokens + measure_tokens + tempo_tokens + pitche_tokens + position_tokens
    max_duration = paras['max_duration'] * (paras['nth'] // 4)
    duration_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(1, max_duration+1))) + [tokenint2str(-max_duration)]
    velocity_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(paras['max_track_number'])))
    instrument_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(129)))

    print(f'Event tokens size: {len(event_tokens)}')
    print(f'- Measure tokens size: {len(measure_tokens)} ({measure_tokens})')
    print(f'- Corpus position tokens size: {len(corpus_position_tokens)}')
    print(f'- Max corpus position: {max(text_to_token(cptoken)[1] for cptoken in corpus_position_tokens)}')
    print(f'- Theoretical largest position = {get_theoretical_max_position(paras["nth"])})')
    print(f'Max measure span size that fits in max_sample_length: {max_fitting_measure_number}', )
    print(f'  (max_sample_length = {max_sample_length}, max measure number in piece = {max_measure_length})')

    print('Duration token size:', len(duration_tokens), duration_tokens)
    print('- Corpus duration tokens size:', len(corpus_duration_tokens))
    print('- Largest possible duration:', paras['max_duration'] * paras['nth'] // 4)
    print('- Largest corpus duration:', max(
        # corpus_duration_tokens, key=lambda x: int(x, TOKEN_INT2STR_BASE)
        tokenstr2int(x) for x in corpus_duration_tokens
    ))

    print('Velocity token size:', len(velocity_tokens), velocity_tokens)
    print('Time:', time()-start_time)

    # build a dictionary for every token list
    vocab_dicts = {
        'paras': paras,
        'max_sample_length': max_sample_length,
        'max_measure_number': max_fitting_measure_number + 1, # plus 1 for head section
        'event_tokens': make_token_dict(event_tokens),
        'duration_tokens': make_token_dict(duration_tokens),
        'velocity_tokens': make_token_dict(velocity_tokens),
        'track_number_tokens': make_token_dict(track_number_tokens),
        'instrument_tokens': make_token_dict(instrument_tokens)
    }
    return vocab_dicts


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--max-sample-length',
        dest='max_sample_length',
        type=int,
        default=4096
    )
    parser.add_argument(
        'input_file_path',
    )
    parser.add_argument(
        'output_dir_path',
    )

    args = parser.parse_args()

    paras, pieces = file_to_paras_and_pieces(args.input_file_path)
    assert len(pieces) > 0, 'empty corpus'

    if not os.path.isdir(args.output_dir_path):
        os.makedirs(args.output_dir_path)

    vocabs = build_vocabs(pieces, paras, args.max_sample_length)
    vocabs_json_path = os.path.join(args.output_dir_path, 'vocabs.json')
    with open(vocabs_json_path, 'w+', encoding='utf8') as vocabs_file:
        json.dump(vocabs, vocabs_file)

    start_time = time()
    print('Begin write npz')
    # serialize pieces into lists of numpy arrays
    # each is processed from a token and has length of seven:
    #   event, duration, velocity, track_numer, instrument, position, measure_number
    # fill each attribute with the index of the text.
    # if a token doesn't have an attribute of some type, fill the index of PAD instead.
    # if a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    # measure_number attribute is dynamically determined in training time, so don't fill it now.
    pieces_numpy_list = [piece_to_numpy(p, vocabs) for p in pieces]
    # if debug:
    # np.savetxt('piece_to_numpy0.txt', pieces_numpy_list[0], fmt='%d')
    np.savez_compressed(os.path.join(args.output_dir_path, 'data.npz'), *pieces_numpy_list)
    print('Time:', time()-start_time)
