import io
import os
from time import time
import yaml

import numpy as np

from .tokens import (
    BEGIN_TOKEN_STR,
    END_TOKEN_STR,
    b36str2int,
    int2b36str,
    get_largest_possible_position,
    PADDING_TOKEN_STR,
    SUPPORTED_TIME_SIGNATURES,
)

def to_paras_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'paras')

def to_corpus_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'corpus')

def to_pathlist_file_path(corpus_dir_path: str) -> str:
    # whaaaaat
    return os.path.join(corpus_dir_path, 'pathlist')

def to_shape_vocab_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'shape_vocab')

def to_vocabs_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'vocabs.json')

def dump_corpus_paras(aras_dict: dict) -> str:
    return yaml.dump(aras_dict)

def get_corpus_paras(corpus_dir_path: str):
    with open(to_paras_file_path(corpus_dir_path), 'r', encoding='utf8') as paras_file:
        corpus_paras_dict = yaml.safe_load(paras_file)
    return corpus_paras_dict

# use iterator to handle large file
class CorpusIterator:
    def __init__(self, corpus_dir_path) -> None:
        self.file_path = os.path.join(corpus_dir_path, 'corpus')
        self.length = None
        self.file = None

    # context manager
    def __enter__(self):
        self.file = open(self.file_path, 'r', encoding='utf8')
        return self

    def __exit__(self, value, type, trackback):
        self.file.close()

    def __len__(self) -> int:
        if self.length is None:
            self.file.seek(0)
            self.length = sum(
                1 if line.startswith('BOS') else 0
                for line in self.file
            )
        return self.length

    def __iter__(self):
        self.file.seek(0)
        for line in self.file:
            if len(line) > 1:
                yield line[:-1] # remove \n at the end


def make_token_dict(token_list: list) -> dict:
    token_dict = {
        'size': len(token_list),
        'id2text': {str(i):t for i, t in enumerate(token_list)}, # in json, key can only be string
        'text2id': {t:i for i, t in enumerate(token_list)},
    }
    return token_dict


def build_vocabs(
        corpus_iterator,
        paras: dict,
        bpe_shapes_list: list):
    """
        Parameters:
        - bpe_shapes_list: The multinote shape are learned by bpe algorithms.
          If bpe is not performed, `bpe_shapes_list` should be empty

        Event tokens include:
        - shape
        - positions in measure
        - measure and time_signature
        - tempo change
        - track and instrument
        - begin-of-score and end-of-score
    """

    start_time = time()

    # shape vacob
    shape_tokens = ['N', 'N~'] + bpe_shapes_list

    # other tokens vocabs
    track_tokens = [
        'R'+int2b36str(i) for i in range(paras['max_track_number'])
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    # tempo_max is inclusive
    tempo_tokens = [
        'T'+int2b36str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]
    if paras['position_method'] == 'event':
        position_tokens = [
            'P'+int2b36str(i) for i in range(get_largest_possible_position(paras['nth']))
        ]
    else:
        position_tokens = []
    measure_tokens = [
        f'M{int2b36str(n)}/{int2b36str(d)}' for n, d in SUPPORTED_TIME_SIGNATURES
    ]

    # place padding token in front so that they have same index across all vocabulary
    event_tokens = ([PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR] + shape_tokens
             + track_tokens + tempo_tokens + position_tokens + measure_tokens)

    max_duration = paras['max_duration']
    pitch_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(128)))
    duration_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(1, max_duration+1)))
    velocity_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(paras['max_track_number'])))
    instrument_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(129)))
    position_attr_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(get_largest_possible_position(paras['nth']))))

    token_count_per_piece = []
    max_measure_length = 0 # max measure number in all pieces
    corpus_measure_tokens = set()
    corpus_position_tokens = set() # we want to see how sparse measure and position is
    for piece in corpus_iterator:
        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))
        measure_indices = [i for i, t in enumerate(text_list) if t[0] == 'M']
        max_measure_length = max(max_measure_length, len(measure_indices))

        measures = {t for t in text_list if t[0] == 'M'}
        corpus_measure_tokens.update(measures)
        if paras['position_method'] == 'event':
            positions = {t[1:] for t in text_list if t[0] == 'P'}
        else:
            positions = {t.split(':')[-1] for t in text_list if t[0] == 'N'}
        corpus_position_tokens.update(positions)
    corpus_measure_tokens = list(corpus_measure_tokens)
    corpus_position_tokens = list(corpus_position_tokens)

    summary_string = (
        f'Average tokens per piece: {sum(token_count_per_piece) / len(token_count_per_piece)}\n'\
        f'Event tokens size: {len(event_tokens)}\n'\
        f'Measure tokens size: {len(measure_tokens)}\n'\
        f'- Corpus measure size: {len(corpus_measure_tokens)} ({corpus_measure_tokens})\n'\
        f'Position token size: {len(position_attr_tokens)}\n'\
        f'- Corpus position tokens size: {len(corpus_position_tokens)}\n'\
        f'- Max corpus position: {max(b36str2int(cpts) for cpts in corpus_position_tokens)}\n'\
        f'Tempo token size: {len(tempo_tokens)}\n'\
        f'Duration size: {len(shape_tokens)}\n'\
        f'- Largest possible duration: {paras["max_duration"] * paras["nth"] // 4}\n'\
        f'Velocity size: {len(velocity_tokens)}\n'\
        f'Vocabulary build time: {time()-start_time}'
    )

    # build a dictionary for every token list
    vocab_dicts = {
        'paras': paras,
        'bpe_iter': len(bpe_shapes_list),
        'padding_token': PADDING_TOKEN_STR,
        'events': make_token_dict(event_tokens),
        'pitchs': make_token_dict(pitch_tokens),
        'durations': make_token_dict(duration_tokens),
        'velocities': make_token_dict(velocity_tokens),
        'track_numbers': make_token_dict(track_number_tokens),
        'instruments': make_token_dict(instrument_tokens),
        'tempos': make_token_dict(tempo_tokens),
        'positions': make_token_dict(position_attr_tokens)
    }

    return vocab_dicts, summary_string


ATTR_NAME_TO_FEATURE_INDEX = {
    'evt': 0, 'events': 0,
    'pit': 1, 'pitchs': 1,
    'dur': 2, 'durations': 2,
    'vel': 3, 'velocities': 3,
    'trn': 4, 'track_numbers': 4,
    'ins': 5, 'instruments': 5,
    'pos': 6, 'positions': 6,
    'tmp': 7, 'tempos': 7,
    'mea': 8, 'measure_numbers': 8,
}


def text_list_to_array(text_list: list, vocabs: dict) -> np.ndarray:
    """
        Serialize pieces into numpy array for the model input.
        Each token is processed into an 9-dimension vector:
            event, pitch, duration, velocity, track_number, instrument, position, tempo, measure_number
        If a token doesn't have an attribute, fill the index of PAD (which should be zero)
        If a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    """
    position_method = vocabs['paras']['position_method']
    event_text2id = vocabs['events']['text2id']
    # in build_vocabs, padding token is made to be the 0th word
    padding = 0
    x = np.full((len(text_list), 9), fill_value=padding, dtype=np.int16)

    tracks_count = 0

    cur_position_id = 0
    cur_measure_number = 0 # measure_number starts at 1, 0 is padding
    cur_tempo_id = padding
    # cur_timesig_id = -1
    for i, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue

        if typename == 'B' or typename == 'E':
            x[i][ATTR_NAME_TO_FEATURE_INDEX['evt']] = event_text2id[text]

        elif typename == 'R':
            event_text, instrument = text.split(':')
            track_number = event_text[1:]
            tracks_count += 1
            x[i][ATTR_NAME_TO_FEATURE_INDEX['evt']] = event_text2id[event_text]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['trn']] = vocabs['track_numbers']['text2id'][track_number]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['ins']] = vocabs['instruments']['text2id'][instrument]

        elif typename == 'M':
            cur_position_id = vocabs['positions']['text2id']['0']
            cur_measure_number += 1
            x[i][ATTR_NAME_TO_FEATURE_INDEX['evt']] = event_text2id[text]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['tmp']] = cur_tempo_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['pos']] = cur_position_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['mea']] = cur_measure_number

        elif typename == 'T':
            event_text, *attr = text = text.split(':')
            cur_tempo_id = vocabs['tempos']['text2id'][event_text]
            if position_method == 'attribute':
                cur_position_id = b36str2int(attr[0])
            x[i][ATTR_NAME_TO_FEATURE_INDEX['evt']] = event_text2id[event_text]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['tmp']] = cur_tempo_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['pos']] = cur_position_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['mea']] = cur_measure_number
            # because tempo come after a position or measure token, which already assigned previous tempo's id
            x[i-1][ATTR_NAME_TO_FEATURE_INDEX['tmp']] = cur_tempo_id

        elif typename == 'P':
            cur_position_id = vocabs['positions']['text2id'][text[1:]]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['evt']] = event_text2id[event_text]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['tmp']] = cur_tempo_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['pos']] = cur_position_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['mea']] = cur_measure_number

        elif typename == 'N' or typename == 'S':
            event_text, *attr = text.split(':')
            if position_method == 'attribute':
                cur_position_id = vocabs['positions']['text2id'][attr[4]]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['evt']] = event_text2id[event_text]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['pit']] = vocabs['pitchs']['text2id'][attr[0]]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['dur']] = vocabs['durations']['text2id'][attr[1]]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['vel']] = vocabs['velocities']['text2id'][attr[2]]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['trn']] = vocabs['track_numbers']['text2id'][attr[3]]
            assert b36str2int(attr[3]) < tracks_count, 'Invalid track number'
            x[i][ATTR_NAME_TO_FEATURE_INDEX['ins']] = x[b36str2int(attr[3])+1, ATTR_NAME_TO_FEATURE_INDEX['ins']]
            x[i][ATTR_NAME_TO_FEATURE_INDEX['tmp']] = cur_tempo_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['pos']] = cur_position_id
            x[i][ATTR_NAME_TO_FEATURE_INDEX['mea']] = cur_measure_number

        else:
            raise ValueError('unknown typename')
    return x


def array_to_text_list(data, vocabs: dict, is_input=True):
    """
        Inverse of text_list_to_array.
        if is_input is True, expect shape of (None, 9)
        otherwise, expect shape (None, 6) if position_method == 'event' else (None, 7)
    """
    position_method = vocabs['paras']['position_method']
    assert len(data.shape) == 2 and data.shape[0] > 1, f'bad numpy array shape: {data.shape}'
    if is_input:
        assert data.shape[1] == 9, f'bad numpy array shape: {data.shape}'
    else:
        assert data.shape[1] == 6 if position_method == 'event' else 7, f'bad numpy array shape: {data.shape}'


    text_list = []
    for i in range(data.shape[0]):
        # in json, key can only be string
        event_text = vocabs['events']['id2text'][str(data[i][ATTR_NAME_TO_FEATURE_INDEX['evt']])]
        typename = event_text[0]

        if typename == 'B' or typename == 'E':
            text_list.append(event_text) # no additional attribute needed

        elif typename == 'R':
            # track token has instrument attribute
            token_text = event_text + ':' + vocabs['instruments']['id2text'][str(data[i][ATTR_NAME_TO_FEATURE_INDEX['ins']])]
            text_list.append(token_text)

        elif typename == 'M':
            text_list.append(event_text)

        elif typename == 'T':
            text_list.append(event_text)

        elif typename == 'P':
            text_list.append(event_text)

        elif typename == 'N' or typename == 'S':
            # subtract one because there is padding token at the beginning of the vocab
            track_number = data[i][ATTR_NAME_TO_FEATURE_INDEX['trn']] - 1
            token_text = (
                event_text + ':'
                + vocabs['pitchs']['id2text'][str(data[i][ATTR_NAME_TO_FEATURE_INDEX['pit']])] + ':'
                + vocabs['durations']['id2text'][str(data[i][ATTR_NAME_TO_FEATURE_INDEX['dur']])] + ':'
                + vocabs['velocities']['id2text'][str(data[i][ATTR_NAME_TO_FEATURE_INDEX['vel']])] + ':'
                + int2b36str(track_number))
            if position_method == 'attribute':
                token_text += ':' + vocabs['positions']['id2text'][str(data[i][ATTR_NAME_TO_FEATURE_INDEX['pos']])]
            text_list.append(token_text)
        elif event_text == PADDING_TOKEN_STR:
            pass
        else:
            raise ValueError(f'unknown typename of event_text: {event_text}')
    # print(track_number_to_event)
    return text_list

def get_input_array_debug_string(input_array: np.ndarray, mps_sep_indices, vocabs: dict):
    array_text_byteio = io.BytesIO()
    np.savetxt(array_text_byteio, input_array, fmt='%d')
    array_savetxt_list = array_text_byteio.getvalue().decode().split('\n')
    reconstructed_text_list = array_to_text_list(input_array, vocabs=vocabs)
    if mps_sep_indices is None:
        mps_sep_indices = []
    debug_str = 'evt  pit  dur  vel  trn  ins  tmp  pos  mea  reconstructed_text\n'
    debug_str += (
        '\n'.join([
            ' '.join([f'{s:>4}' for s in a.split()])
            + f'  {z:<16}' + (' MPS_SEP' if i in mps_sep_indices else '')
            for i, (a, z) in enumerate(zip(array_savetxt_list, reconstructed_text_list))
        ])
    )
    if isinstance(mps_sep_indices, np.ndarray):
        debug_str += '\nmps_sep_indices: ' + str(mps_sep_indices)
    return debug_str
