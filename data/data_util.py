import numpy as np
from time import time
from tokens import *

def make_token_dict(token_list: list) -> dict:
    token_dict = {
        'size': len(token_list),
        'id2text': {i:t for i, t in enumerate(token_list)},
        'text2id': {t:i for i, t in enumerate(token_list)},
    }
    return token_dict


def build_vocabs(piece_iterator, paras: dict, max_sample_length: str):
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

    # max measure number in all pieces
    max_measure_length = 0
    # the most number of measures span that can fit in the size of max_sample_length
    max_fitting_measure_number = 0
    corpus_measure_tokens = set()
    corpus_position_tokens = set()
    corpus_duration_tokens = set()
    for piece in piece_iterator:
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
        durations = {t.split(':')[1] for t in text_list if t[0] == 'N'}
        corpus_measure_tokens.update(measures)
        corpus_position_tokens.update(positions)
        corpus_duration_tokens.update(durations)
    corpus_measure_tokens = list(corpus_measure_tokens)
    corpus_position_tokens = list(corpus_position_tokens)
    corpus_duration_tokens = list(corpus_duration_tokens)

    # place special token in front so that they have same index across all vocabulary
    event_tokens = SPECIAL_TOKENS + ['BOS', 'EOS'] + track_tokens + measure_tokens + pitche_tokens + position_tokens

    if paras['tempo_method'].endswith('event'):
        tempo_event_tokens = ['T'+tokenint2str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)]
        event_tokens += tempo_event_tokens

    max_duration = paras['max_duration'] * (paras['nth'] // 4)
    duration_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(1, max_duration+1))) + [tokenint2str(-max_duration)]
    velocity_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(paras['max_track_number'])))
    instrument_tokens = SPECIAL_TOKENS + list(map(tokenint2str, range(129)))
    tempo_attribute_tokens = SPECIAL_TOKENS + tempo_attribute_tokens

    summary_string = (
        f'Event tokens size: {len(event_tokens)}\n'
        + f'Measure tokens size: {len(measure_tokens)}\n'
        + f'- Corpus measure size: {len(corpus_measure_tokens)} ({corpus_measure_tokens})\n'
        + f'Position token size: {len(position_tokens)}\n'
        + f'- Corpus position tokens size: {len(corpus_position_tokens)}\n'
        + f'- Max corpus position: {max(text_to_attrs(cptoken)[1][0] for cptoken in corpus_position_tokens)}\n'
        + f'Max measure number that fits max_sample_length: {max_fitting_measure_number}\n'
        + f'  (max_sample_length = {max_sample_length}, max measure number in piece = {max_measure_length})\n'
        + (f'Tempo token size: {len(tempo_event_tokens)}\n' if tempo_event_tokens else '')
        + f'Duration token size: {len(duration_tokens)}\n'
        + f'- Corpus duration tokens size: {len(corpus_duration_tokens)}\n'
        + f'- Largest possible duration: {paras["max_duration"] * paras["nth"] // 4}\n'
        + f'- Largest corpus duration: {max(tokenstr2int(x) for x in corpus_duration_tokens)}\n'
        + f'Velocity token size: {len(velocity_tokens)}\n'
        + f'Tempo attribute size: {len(tempo_attribute_tokens)}\n'
        + f'Vocabulary build time: {time()-start_time}'
    )

    # build a dictionary for every token list
    vocab_dicts = {
        'paras': paras,
        'max_sample_length': max_sample_length,
        'max_measure_number': max_fitting_measure_number + 1, # plus 1 for head section
        'events': make_token_dict(event_tokens),
        'durations': make_token_dict(duration_tokens),
        'velocities': make_token_dict(velocity_tokens),
        'track_numbers': make_token_dict(track_number_tokens),
        'instruments': make_token_dict(instrument_tokens),
        'tempos': make_token_dict(tempo_attribute_tokens),
    }
    return vocab_dicts, summary_string


def text_list_to_numpy(text_list: str, vocabs: dict) -> np.ndarray:
    """
        Serialize pieces into of numpy arrays.
        Each token is processed into an 8-dimension vector:
            event, duration, velocity, track_number, instrument, tempo, position, measure_number
        If a token doesn't have an attribute, fill the index of PAD.
        If a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    """
    x = np.full((len(text_list), 8), fill_value=-1, dtype=np.int16)

    event_text2id = vocabs['events']['text2id']
    # d_pad_id = vocabs['durations']['text2id']['[PAD]']
    # v_pad_id = vocabs['velocities']['text2id']['[PAD]']
    # r_pad_id = vocabs['track_numbers']['text2id']['[PAD]']
    # i_pad_id = vocabs['instruments']['text2id']['[PAD]']
    # t_pad_id = vocabs['tempos']['text2id']['[PAD]']

    # in build_vocabs, [PAD] is made to be 0
    d_pad_id = 0
    v_pad_id = 0
    r_pad_id = 0
    i_pad_id = 0
    t_pad_id = 0

    # position and measure use pure integer instead of id
    # head section and eos are all measure=0, position=0
    # measure index begin at 1
    cur_position = 0
    cur_measure_number = 0
    cur_tempo_id = -1
    for i, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue

        if typename == 'B' or typename == 'E':
            x[i] = [event_text2id[text], d_pad_id, v_pad_id, r_pad_id, i_pad_id, t_pad_id, 0, 0]

        elif typename == 'R':
            event_text, instrument = text.split(':')
            track_number = event_text[1:]
            x[i] = [
                event_text2id[event_text], d_pad_id, v_pad_id, vocabs['track_numbers']['text2id'][track_number],
                vocabs['instruments']['text2id'][instrument], t_pad_id, 0, 0
            ]

        elif typename == 'M':
            cur_position = 0
            cur_measure_number += 1
            if '+' in text:
                tempo_text = text.split('+')[1]
                cur_tempo_id = vocabs['tempos']['text2id'][tempo_text]
            x[i] = [
                event_text2id[text], d_pad_id, v_pad_id, r_pad_id,
                i_pad_id, t_pad_id, cur_position, cur_measure_number
            ]

        elif typename == 'T':
            cur_tempo_id = vocabs['tempos']['text2id'][text[1:]]
            x[i] = [
                event_text2id[text], d_pad_id, v_pad_id, r_pad_id,
                i_pad_id, cur_tempo_id, cur_position, cur_measure_number
            ]

        elif typename == 'P':
            if '+' in text:
                event_text, tempo_text = text.split('+')
                cur_position = tokenstr2int(event_text[1:])
                cur_tempo_id = vocabs['tempos']['text2id'][tempo_text]
                x[i] = [
                    event_text2id[text], d_pad_id, v_pad_id, r_pad_id,
                    i_pad_id, cur_tempo_id, cur_position, cur_measure_number
                ]
            else:
                cur_position = tokenstr2int(text[1:])
                x[i] = [
                    event_text2id[text], d_pad_id, v_pad_id, r_pad_id,
                    i_pad_id, cur_tempo_id, cur_position, cur_measure_number
                ]

        elif typename == 'N':
            event_text, *attr = text.split(':')
            x[i] = [
                event_text2id[event_text],
                vocabs['durations']['text2id'][attr[0]],
                vocabs['velocities']['text2id'][attr[1]],
                vocabs['track_numbers']['text2id'][attr[2]],
                # get instrument id from track token
                x[tokenstr2int(attr[2])+1, 4],
                cur_tempo_id,
                cur_position,
                cur_measure_number
            ]

        else:
            raise ValueError('unknown typename')
    return x


def numpy_to_text_list(data: np.ndarray, vocabs: dict, tempo_method: str):
    # event, duration, velocity, track_number, instrument, tempo, position, measure_number
    assert data.shape[0] > 2 and data.shape[1] == 8, f'bad numpy array shape: {data.shape}'

    text_list = []
    for i in range(data.shape[0]):
        event_text = vocabs['events']['id2text'][data[i][0]]
        typename = event_text[0]

        if typename == 'B' or typename == 'E':
            text_list.append(event_text) # no additional attribute needed

        elif typename == 'R':
            # track token has instrument attribute
            token_text = event_text + ':' + vocabs['instruments']['id2text'][data[i][4]]
            text_list.append(token_text)

        elif typename == 'M':
            text_list.append(event_text)

        elif typename == 'T':
            text_list.append(event_text)

        elif typename == 'P':
            text_list.append(event_text)

        elif typename == 'N':
            token_text = event_text + ':' \
                + vocabs['durations']['id2text'][data[i][1]] + ':' \
                + vocabs['velocities']['id2text'][data[i][2]] + ':' \
                + vocabs['track_numbers']['id2text'][data[i][3]]
            text_list.append(token_text)

        else:
            raise ValueError('unknown typename')

    return text_list
