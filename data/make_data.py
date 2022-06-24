import json
import numpy as np
import os
import random
import re
from argparse import ArgumentParser
from time import time

from midi_util import parse_para_yaml
from tokens import *


def piece_2_numpy(piece: str, vocabs: dict) -> np.ndarray:
    text_list = piece.split(' ')
    x = np.full((len(text_list), 7), fill_value=vocabs['event_tokens']['text2id']['PAD'], dtype=np.int16)
    # event, duration, velocity, track_numer, instrument, position, measure_number
    cur_position = 0
    for i, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue
        if typename == 'R':
            event_text = text.split(':')[0]
            x[i, 0] = vocabs['event_tokens']['text2id'][event_text]
            x[i, 3] = int(event_text[1:], TOKEN_UINT2STR_BASE)
            x[i, 4] = int(text.split(':')[1], TOKEN_UINT2STR_BASE)
        elif typename == 'M':
            x[i, 0] = vocabs['event_tokens']['text2id']['M']
        else:
            event_text, *attr = text.split(':')
            try:
                x[i, 0] = vocabs['event_tokens']['text2id'][event_text]
            except IndexError:
                x[i, 0] = vocabs['event_tokens']['text2id']['UNK']
            if len(attr) > 0:
                for j, a in enumerate(attr):
                    x[i, j+1] = int(a, TOKEN_UINT2STR_BASE)
            if typename == 'P':
                cur_position = int(text[1:], TOKEN_UINT2STR_BASE)
                x[i, 5] = cur_position
            if typename == 'N':
                x[i, 5] = cur_position
    return x


def make_token_dict(token_list: list) -> dict:
    token_dict = {
        'id2text': {i:t for i, t in enumerate(token_list)},
        'text2id': {t:i for i, t in enumerate(token_list)},
        'size': len(token_list),
    }
    return token_dict


def build_vocabs(body: str, paras: dict, max_sample_length: str):
    start_time = time()
    # Event tokens include:
    #   note pitches, positions in measure, end-of-position, measure numbers, time signatures,
    #   tempos, track, begin-of-score and end-of-score
    # pitches, positions and tempos are determined by arguments
    # although some positions may never appear in the corpus, which could be discarded
    # time signature is sampled from corpus
    # measure number size = max amount of measures that a max_sample_length can fit in
    pitche_tokens = [
        'N'+uint2str(i) for i in range(128)
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    tempo_tokens = [
        'T'+uint2str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]
    track_token = [
        'R'+uint2str(i) for i in range(paras['max_track_number'])
    ]
    measure_token = ['M']

    # max measure number in all pieces
    max_measure_length = 0
    # the most number of measures span that can fit in the size of max_sample_length
    max_fitting_measure_number = 0 
    times_sig_tokens = set()
    position_tokens = set()
    corpus_duration_tokens = set()
    piece_list = body.split('\n')
    for piece in piece_list:
        if len(piece) == 0:
            continue
        text_list = piece.split(' ')

        measure_indices = [i for i, t in enumerate(text_list) if t == 'M']
        max_measure_length = max(max_measure_length, len(measure_indices))
        cur_piece_fitting_measure_number = 0
        for size in range(len(measure_indices)-1, 1, -1):
            for start in range(0, len(measure_indices)-size):
                if measure_indices[start+size] - measure_indices[start] < max_sample_length:
                    cur_piece_fitting_measure_number = size
                    break
            if cur_piece_fitting_measure_number > 0:
                break
        max_fitting_measure_number = max(
            max_fitting_measure_number,
            cur_piece_fitting_measure_number
        )

        time_sigs = {t for t in text_list if t[0] == 'S'}
        positions = {t for t in text_list if t[0] == 'P'}
        durations = {t.split(':')[1] for t in text_list if t[0] == 'N'}
        times_sig_tokens.update(time_sigs)
        position_tokens.update(positions)
        corpus_duration_tokens.update(durations)
    position_tokens = list(position_tokens)
    times_sig_tokens = list(times_sig_tokens)
    corpus_duration_tokens = list(corpus_duration_tokens)

    event_tokens = SPECIAL_TOKEN_TEXTS + pitche_tokens + position_tokens + tempo_tokens + times_sig_tokens + track_token + measure_token

    print(f'Event tokens size: {len(event_tokens)}')
    print(f'- Times sig tokens number: {len(times_sig_tokens)} ({times_sig_tokens})')
    print(f'- Corpus position tokens number: {len(position_tokens)}. Max corpus position: {max(text_2_token(ptoken)[1] for ptoken in position_tokens)}')
    print(f'  (theoretical largest position = {31 * (paras["nth"] // 16)})')
    print(f'Max measure span size that fits in max_sample_length: {max_fitting_measure_number}')
    print(f'  (max sample slength = {max_sample_length}, max measure number in piece = {max_measure_length})')
    print(f'Time: {time()-start_time}')

    # Note attribute tokens are durations, velocities, track numbers and instruments
    # all of them are determined by arguments
    # but duration could be sparse due to large nth
    duration_tokens = list(map(uint2str, range(paras['nth'] * paras['max_duration'])))
    velocity_tokens = list(map(uint2str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = list(map(uint2str, range(paras['max_track_number'])))
    instrument_tokens = list(map(uint2str, range(129)))
    duration_tokens.sort()
    velocity_tokens.sort()
    track_number_tokens.sort()
    instrument_tokens.sort()

    print('Theoretical duration tokens size:', paras['max_duration'] * paras['nth'] // 4)
    print('Corpus duration tokens size:', len(corpus_duration_tokens))
    print('Largest corpus duration:', max(
        # corpus_duration_tokens, key=lambda x: int(x, TOKEN_UINT2STR_BASE)
        int(x, TOKEN_UINT2STR_BASE) for x in corpus_duration_tokens
    ))
    print('Velocity token size:', len(velocity_tokens))
    print('Track tokens size:', paras['max_track_number'])
    print('Instrument tokens size: 129')

    # build a dictionary for every token list
    vocab_dicts = {
        'paras': paras,
        'max_sample_length': max_sample_length,
        'max_measure_number': max_fitting_measure_number,
        'event_tokens': make_token_dict(event_tokens),
        'duration_tokens': make_token_dict(duration_tokens),
        'velocitie_tokens': make_token_dict(velocity_tokens),
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
        '--split-ratio',
        nargs=3,
        dest='split_ratio',
        type=float,
        default=[8, 1, 1],
        metavar=('TRAIN', 'TEST', 'VALID')
    )
    parser.add_argument(
        'input_file_path',
    )
    parser.add_argument(
        'output_dir_path',
    )

    args = parser.parse_args()

    with open(args.input_file_path, 'r', encoding='utf8') as corpus_file:
        # get settings and main texts
        texts = corpus_file.read()
        _, head, body = texts.split('---')
        paras = parse_para_yaml(head)
        print(paras)
    body = body.lstrip('\n').rstrip('\n')

    if not os.path.isdir(args.output_dir_path):
        os.makedirs(args.output_dir_path)

    vocabs = build_vocabs(body, paras, args.max_sample_length)
    vocabs_json_path = os.path.join(args.output_dir_path, 'vocabs.json')
    with open(vocabs_json_path, 'w+', encoding='utf8') as vocabs_file:
        json.dump(vocabs, vocabs_file)

    # split body texts into train, test and valid
    pieces = body.split('\n')
    random.shuffle(pieces)
    train_split_size = int(len(pieces)*args.split_ratio[0]/sum(args.split_ratio))
    test_split_size  = int(len(pieces)*args.split_ratio[1]/sum(args.split_ratio))
    valid_split_size = len(pieces) - train_split_size - test_split_size
    print('Total pieces:', len(pieces))
    print('Train split size', train_split_size)
    print('Test split size', test_split_size)
    print('Valid split size', valid_split_size)
    train_pieces = pieces[:train_split_size]
    test_pieces = pieces[train_split_size:train_split_size+test_split_size]
    valid_pieces = pieces[train_split_size+test_split_size:]

    # serialize each splits into lists of numpy arrays
    # each is processed from a token and has length of seven:
    #   event, duration, velocity, track_numer, instrument, position, measure_number
    # fill each attribute with the index of the text.
    # if a token doesn't have an attribute of some type, fill the index of PAD instead.
    # if a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    # measure_number attribute is dynamically determined in training time, so don't fill it now.

    train_pieces_numpy = np.array([piece_2_numpy(p, vocabs) for p in train_pieces], dtype=object)
    test_pieces_numpy  = np.array([piece_2_numpy(p, vocabs) for p in test_pieces], dtype=object)
    valid_pieces_numpy = np.array([piece_2_numpy(p, vocabs) for p in valid_pieces], dtype=object)

    np.save(os.path.join(args.output_dir_path, 'train.npy'), train_pieces_numpy, allow_pickle=True)
    np.save(os.path.join(args.output_dir_path, 'test.npy'), test_pieces_numpy, allow_pickle=True)
    np.save(os.path.join(args.output_dir_path, 'valid.npy'), valid_pieces_numpy, allow_pickle=True)
