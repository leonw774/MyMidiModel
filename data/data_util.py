import json
import os
import random
import subprocess
from time import time
from tqdm import tqdm

import numpy as np

from .tokens import (
    tokenstr2int,
    tokenint2str,
    get_largest_possible_position,
    SPECIAL_TOKENS,
    TOKEN_INT2STR_BASE,
    SUPPORTED_TIME_SIGNATURES,
)


def do_bpe(piece_iterator, paras: dict, bpe_iter: int, corpus_file_path: str, result_file_path: str) -> str:
    # Shape format:
    #     'S' (rel_onset ',' rel_pitch ',' rel_duration (is_coutinuing?'~') ';')+ ':' duration_unit ':' velocity ':' track_num
    #
    # When shape contain only one note, it must be:
    #     "S0,0,1;:" base_pitch ':' duration_unit ':' velocity ':' track_num
    # If the note is "continuing", i.e. not ending, due to duration limit,
    # we express it with negtive rel_dur "S0,0,1~;"
    #     "S0,0,1~;:" base_pitch ':' duration_unit ':' velocity ':' track_num
    # These two basic shape can be simplified into Note format:
    #     'N' ':' pitch ':' duration ':' velocity ':' track_num
    # and 'N~' ':' pitch ':' duration ':' velocity ':' track_num
    # with the substitution of "S0,0,1" to "N" and "S0,0,-1" to "N~"

    nth = paras['nth']
    shape_vocabs_file_path = corpus_file_path+'_vocabs'
    multinotes_file_path = corpus_file_path+'_multinotes'

    bpe_program_path = './bpe/bpe'

    # check bpe shape vocab file
    if not os.path.isfile(shape_vocabs_file_path):
        raise FileNotFoundError(f'Cannot find vocabs output file: {shape_vocabs_file_path}')

    # get shape_tokens
    with open(shape_vocabs_file_path, 'r', encoding='utf8') as vocabs_file:
        bpe_shapes_list = vocabs_file.read().splitlines()

    # check bpe encoded multinote text file
    if not os.path.isfile(multinotes_file_path):
        raise FileNotFoundError(f'Cannot find multinotes output file: {multinotes_file_path}')

    # rebuild text midi now with shapes
    print('Rebuild text corpus now with shape vocabs')
    token_number_list = []
    with open(result_file_path, 'a', encoding='utf8') as bpe_result_file:
        with open(multinotes_file_path, 'r', encoding='utf8') as mn_file:
            for piece in tqdm(piece_iterator, total=len(piece_iterator)):
                mnline = next(mn_file)
                mn_list = mnline.split()
                piece_time_struct_list = [t for t in piece.split(' ') if t[0] != 'N']
                token_number_list.append(len(mn_list) + len(piece_time_struct_list))

                result_str = ""
                piece_cursor = 0
                piece_cur_onset = 0
                piece_cur_measure_onset = 0
                piece_cur_measure_length = 0
                added_last_type = 'B'
                mn_cursor = 0
                mn_cur_onset = int(mn_list[0].split('@')[0], TOKEN_INT2STR_BASE)
                while (piece_cursor < len(piece_time_struct_list)) or (mn_cursor < len(mn_list)):
                    if piece_cur_onset <= mn_cur_onset:
                        if not (added_last_type == 'P' and piece_time_struct_list[piece_cursor][0] == 'P'):
                            result_str += piece_time_struct_list[piece_cursor] + ' '
                        else:
                            token_number_list[-1] -= 1
                        added_last_type = piece_time_struct_list[piece_cursor][0]
                        piece_cursor += 1
                        if piece_cursor >= len(piece_time_struct_list):
                            continue
                        # update
                        text = piece_time_struct_list[piece_cursor]
                        if text[0] == 'M':
                            piece_cur_measure_onset += piece_cur_measure_length
                            piece_cur_onset = piece_cur_measure_onset
                            n, d = (int(x, TOKEN_INT2STR_BASE) for x in text[1:].split('/'))
                            piece_cur_measure_length = n * nth // d
                        elif text[0] == 'P':
                            piece_cur_onset = piece_cur_measure_onset + int(text[1:], TOKEN_INT2STR_BASE)
                        elif text[0] == 'E':
                            piece_cur_onset += piece_cur_measure_length
                    else:
                        mn_text = mn_list[mn_cursor].split('@')[1]
                        if mn_text[0] == 'S':
                            shape_text = bpe_shapes_list[int(mn_text[1:mn_text.find(':')], TOKEN_INT2STR_BASE)]
                            mn_text = shape_text + mn_text[mn_text.find(':'):]
                        result_str += mn_text + ' '
                        added_last_type = 'N'
                        mn_cursor += 1
                        if mn_cursor >= len(mn_list):
                            mn_cur_onset = float('inf')
                        else:
                            mn_cur_onset = int(mn_list[mn_cursor].split('@')[0], TOKEN_INT2STR_BASE)

                bpe_result_file.write(result_str[:-1]+'\n') # replace the last space with newline

    print('End rebuild')
    print('Avg token per piece:', sum(token_number_list) / len(token_number_list))
    # os.remove(os.path.join('bpe', shape_vocabs_file_path))
    # os.remove(os.path.join('bpe', multinotes_file_path))
    # print(bpe_shapes_list)
    return bpe_shapes_list


def make_token_dict(token_list: list) -> dict:
    token_dict = {
        'size': len(token_list),
        'id2text': {str(i):t for i, t in enumerate(token_list)}, # in json, key can only be string
        'text2id': {t:i for i, t in enumerate(token_list)},
    }
    return token_dict


def build_vocabs(
        piece_iterator,
        paras: dict,
        output_dir_path: str,
        max_sample_length: str,
        bpe_shapes_list:list):
    start_time = time()
    # Event tokens include:
    #   note (single note), shape (multiple notes), positions in measure, measure/time_signature, tempo change, 
    #   track info, begin-of-score and end-of-score
    # measure number size = max amount of measures that a max_sample_length can fit in

    # shape vacob
    shape_tokens = ['N', 'N~'] + bpe_shapes_list

    # other tokens vocabs
    track_tokens = [
        'R'+tokenint2str(i) for i in range(paras['max_track_number'])
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    # tempo_max is inclusive
    tempo_attribute_tokens = list(map(tokenint2str, range(tempo_min, tempo_max+tempo_step, tempo_step)))
    position_tokens = [
        'P'+tokenint2str(i) for i in range(get_largest_possible_position(paras['nth']))
    ]
    measure_tokens = [
        f'M{tokenint2str(n)}/{tokenint2str(d)}' for n, d in SUPPORTED_TIME_SIGNATURES
    ]
    if paras['tempo_method'] == 'position_attribute':
        position_tokens = [
            position_token+'+'+tempo_attribute
            for position_token in position_tokens
            for tempo_attribute in tempo_attribute_tokens
        ]
    elif paras['tempo_method'] == 'measure_attribute':
        measure_tokens = [
            measure_token+'+'+tempo_attribute
            for measure_token in measure_tokens
            for tempo_attribute in tempo_attribute_tokens
        ]

    # place special token in front so that they have same index across all vocabulary
    event_tokens = SPECIAL_TOKENS + ['BOS', 'EOS'] + shape_tokens + track_tokens + measure_tokens + position_tokens

    if paras['tempo_method'].endswith('event'):
        tempo_event_tokens = ['T'+tokenint2str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)]
        event_tokens += tempo_event_tokens

    max_duration = paras['max_duration'] * (paras['nth'] // 4)
    pitch_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(128)))
    duration_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(1, max_duration+1))) + [tokenint2str(-max_duration)]
    velocity_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(paras['max_track_number'])))
    instrument_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(129)))
    tempo_attribute_tokens = SPECIAL_TOKENS + tempo_attribute_tokens

    # max measure number in all pieces
    max_measure_length = 0
    # the most number of measures span that can fit in the size of max_sample_length
    max_fitting_measure_number = 0
    corpus_measure_tokens = set()
    corpus_position_tokens = set()
    for i, piece in enumerate(piece_iterator):
        text_list = piece.split(' ')
        measure_indices = [i for i, t in enumerate(text_list) if t[0] == 'M']
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
        corpus_measure_tokens.update(measures)
        corpus_position_tokens.update(positions)
    corpus_measure_tokens = list(corpus_measure_tokens)
    corpus_position_tokens = list(corpus_position_tokens)

    summary_string = (
        f'Event tokens size: {len(event_tokens)}\n'
        + f'Measure tokens size: {len(measure_tokens)}\n'
        + f'- Corpus measure size: {len(corpus_measure_tokens)} ({corpus_measure_tokens})\n'
        + f'Position token size: {len(position_tokens)}\n'
        + f'- Corpus position tokens size: {len(corpus_position_tokens)}\n'
        + f'- Max corpus position: {max(int(cptoken[1:].split("+")[0], TOKEN_INT2STR_BASE) for cptoken in corpus_position_tokens)}\n'
        + f'Max measure number that fits max_sample_length: {max_fitting_measure_number}\n'
        + f'  (max_sample_length = {max_sample_length}, max measure number in piece = {max_measure_length})\n'
        + (f'Tempo token size: {len(tempo_event_tokens)}\n' if tempo_event_tokens else '')
        + f'Duration token size: {len(shape_tokens)}\n'
        + f'- Largest possible duration: {paras["max_duration"] * paras["nth"] // 4}\n'
        + f'Velocity token size: {len(velocity_tokens)}\n'
        + f'Tempo attribute size: {len(tempo_attribute_tokens)}\n'
        + f'Vocabulary build time: {time()-start_time}'
    )

    # build a dictionary for every token list
    vocab_dicts = {
        'paras': paras,
        'use_bpe': 1 if len(bpe_shapes_list) > 0 else 0,
        'max_sample_length': max_sample_length,
        'special_tokens': SPECIAL_TOKENS,
        'max_measure_number': max_fitting_measure_number + 1, # plus 1 for head section
        'events': make_token_dict(event_tokens),
        'pitches': make_token_dict(pitch_tokens),
        'durations': make_token_dict(duration_tokens),
        'velocities': make_token_dict(velocity_tokens),
        'track_numbers': make_token_dict(track_number_tokens),
        'instruments': make_token_dict(instrument_tokens),
        'tempos': make_token_dict(tempo_attribute_tokens),
    }

    vocabs_json_path = os.path.join(output_dir_path, 'vocabs.json')
    with open(vocabs_json_path, 'w+', encoding='utf8') as vocabs_file:
        json.dump(vocab_dicts, vocabs_file)

    return vocab_dicts, summary_string


def text_list_to_numpy(text_list: str, vocabs: dict) -> np.ndarray:
    """
        Serialize pieces into of numpy arrays.
        Each token is processed into an 9-dimension vector:
            event, pitch, duration, velocity, track_number, instrument, tempo, position, measure_number
        If a token doesn't have an attribute, fill the index of PAD (which should be zero)
        If a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    """
    x = np.full((len(text_list), 9), fill_value=-1, dtype=np.uint16)

    event_text2id = vocabs['events']['text2id']
    # d_pad_id = vocabs['durations']['text2id']['[PAD]']
    # v_pad_id = vocabs['velocities']['text2id']['[PAD]']
    # r_pad_id = vocabs['track_numbers']['text2id']['[PAD]']
    # i_pad_id = vocabs['instruments']['text2id']['[PAD]']
    # t_pad_id = vocabs['tempos']['text2id']['[PAD]']

    # in build_vocabs, [PAD] is made to be 0
    p_pad_id = 0
    d_pad_id = 0
    v_pad_id = 0
    r_pad_id = 0
    i_pad_id = 0
    t_pad_id = 0

    # position and measure use pure integer instead of id
    # head section and eos are measure=0, position=0
    # measure index begin at 1
    cur_position = 0
    cur_measure_number = 0
    cur_tempo_id = -1
    # cur_timesig_id = -1
    for i, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue

        if typename == 'B' or typename == 'E':
            x[i] = [event_text2id[text], p_pad_id, d_pad_id, v_pad_id, r_pad_id, i_pad_id, t_pad_id, 0, 0]

        elif typename == 'R':
            event_text, instrument = text.split(':')
            track_number = event_text[1:]
            x[i] = [
                event_text2id[event_text], p_pad_id, d_pad_id, v_pad_id, vocabs['track_numbers']['text2id'][track_number],
                vocabs['instruments']['text2id'][instrument], t_pad_id, 0, 0
            ]

        elif typename == 'M':
            cur_position = 0
            cur_measure_number += 1
            if '+' in text:
                tempo_text = text.split('+')[1]
                cur_tempo_id = vocabs['tempos']['text2id'][tempo_text]
            x[i] = [
                event_text2id[text], p_pad_id, d_pad_id, v_pad_id, r_pad_id,
                i_pad_id, t_pad_id, cur_position, cur_measure_number
            ]

        elif typename == 'T':
            cur_tempo_id = vocabs['tempos']['text2id'][text[1:]]
            x[i] = [
                event_text2id[text], p_pad_id, d_pad_id, v_pad_id, r_pad_id,
                i_pad_id, cur_tempo_id, cur_position, cur_measure_number
            ]

        elif typename == 'P':
            if '+' in text:
                event_text, tempo_text = text.split('+')
                cur_position = int(event_text[1:], TOKEN_INT2STR_BASE)
                cur_tempo_id = vocabs['tempos']['text2id'][tempo_text]
                x[i] = [
                    event_text2id[text], p_pad_id, d_pad_id, v_pad_id, r_pad_id,
                    i_pad_id, cur_tempo_id, cur_position, cur_measure_number
                ]
            else:
                cur_position = int(text[1:], TOKEN_INT2STR_BASE)
                x[i] = [
                    event_text2id[text], p_pad_id, d_pad_id, v_pad_id, r_pad_id,
                    i_pad_id, cur_tempo_id, cur_position, cur_measure_number
                ]

        elif typename == 'N' or typename == 'S':
            event_text, *attr = text.split(':')
            x[i] = [
                event_text2id[event_text],
                vocabs['pitches']['text2id'][attr[0]],
                vocabs['durations']['text2id'][attr[1]],
                vocabs['velocities']['text2id'][attr[2]],
                vocabs['track_numbers']['text2id'][attr[3]],
                # get instrument id from track token
                x[tokenstr2int(attr[3])+1, 5],
                cur_tempo_id,
                cur_position,
                cur_measure_number
            ]

        else:
            raise ValueError('unknown typename')
    return x


def numpy_to_text_list(data: np.ndarray, vocabs: dict):
    # event, pitch, duration, velocity, track_number, instrument, tempo, position, measure_number
    assert data.shape[0] > 1 and data.shape[1] == 9, f'bad numpy array shape: {data.shape}'

    track_number_to_event = dict()

    text_list = []
    for i in range(data.shape[0]):
        # in json, key can only be string
        event_text = vocabs['events']['id2text'][str(data[i][0])]
        typename = event_text[0]

        if typename == 'B' or typename == 'E':
            text_list.append(event_text) # no additional attribute needed

        elif typename == 'R':
            # track token has instrument attribute
            track_number_to_event[data[i][4]] = event_text[1:]
            token_text = event_text + ':' + vocabs['instruments']['id2text'][str(data[i][5])]
            text_list.append(token_text)

        elif typename == 'M':
            text_list.append(event_text)

        elif typename == 'T':
            text_list.append(event_text)

        elif typename == 'P':
            text_list.append(event_text)

        elif typename == 'N' or typename == 'S':
            corresponded_track_event = track_number_to_event[data[i][4]]
            token_text = (
                event_text + ':'
                + vocabs['pitches']['id2text'][str(data[i][1])] + ':'
                + vocabs['durations']['id2text'][str(data[i][2])] + ':'
                + vocabs['velocities']['id2text'][str(data[i][3])] + ':'
                + corresponded_track_event)
            text_list.append(token_text)
        elif event_text in SPECIAL_TOKENS:
            pass
        else:
            raise ValueError(f'unknown typename of event_text: {event_text}')
    # print(track_number_to_event)
    return text_list
