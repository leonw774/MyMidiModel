import io
import json
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

class Vocabs:

    class AttrVocab:
        size: int
        id2text: dict
        text2id: dict
        def __init__(self, init_obj) -> None:
            if isinstance(init_obj, list):
                self.size = len(init_obj)
                self.id2text = {i:t for i, t in enumerate(init_obj)}
                self.text2id = {t:i for i, t in enumerate(init_obj)}
            elif isinstance(init_obj, dict):
                self.size = init_obj['size']
                self.id2text = {int(k): v for k, v in init_obj['id2text'].items()} # key are string in json
                self.text2id = init_obj['text2id']
            else:
                raise TypeError(f'AttrVocab\'s init_obj can only be list or dict. Got {init_obj}')
        def __len__(self) -> int:
            return self.size

    paras: dict
    bpe_shapes_list: list
    padding_token: str
    events: AttrVocab
    pitchs: AttrVocab
    durations: AttrVocab
    velocities: AttrVocab
    track_numbers: AttrVocab
    instruments: AttrVocab
    positions: AttrVocab
    tempos: AttrVocab

    def __init__(self, paras: dict, bpe_shapes_list: list, padding_token: str,
            events, pitchs, durations, velocities, track_numbers, instruments, positions, tempos, time_signatures) -> None:
        self.paras = paras
        self.bpe_shapes_list = bpe_shapes_list
        self.padding_token = padding_token
        self.events = Vocabs.AttrVocab(events)
        self.pitchs = Vocabs.AttrVocab(pitchs)
        self.durations = Vocabs.AttrVocab(durations)
        self.velocities = Vocabs.AttrVocab(velocities)
        self.track_numbers = Vocabs.AttrVocab(track_numbers)
        self.instruments = Vocabs.AttrVocab(instruments)
        self.positions = Vocabs.AttrVocab(positions)
        self.tempos = Vocabs.AttrVocab(tempos)
        self.time_signatures = Vocabs.AttrVocab(time_signatures)

    @classmethod
    def from_dict(cls, d):
        return cls(
            d['paras'], d['bpe_shapes_list'], d['padding_token'],
            d['events'], d['pitchs'], d['durations'],
            d['velocities'], d['track_numbers'], d['instruments'],
            d['positions'], d['tempos'], d['time_signatures']
        )

    def to_dict(self) -> dict:
        return {
            'paras': self.paras,
            'bpe_shapes_list': self.bpe_shapes_list,
            'padding_token': self.padding_token,
            'events': vars(self.events),
            'pitchs': vars(self.pitchs),
            'durations': vars(self.durations),
            'velocities': vars(self.velocities),
            'track_numbers': vars(self.track_numbers),
            'instruments': vars(self.instruments),
            'positions': vars(self.positions),
            'tempos': vars(self.tempos),
            'time_signatures': vars(self.time_signatures),
        }


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

def get_corpus_paras(corpus_dir_path: str) -> dict:
    with open(to_paras_file_path(corpus_dir_path), 'r', encoding='utf8') as paras_file:
        corpus_paras_dict = yaml.safe_load(paras_file)
    return corpus_paras_dict

def get_corpus_vocabs(corpus_dir_path: str) -> Vocabs:
    with open(to_vocabs_file_path(corpus_dir_path), 'r', encoding='utf8') as vocabs_file:
        corpus_vocabs_dict = json.load(vocabs_file)
    corpus_vocabs = Vocabs.from_dict(corpus_vocabs_dict)
    return corpus_vocabs

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

def build_vocabs(
        corpus_iterator,
        paras: dict,
        bpe_shapes_list: list):
    """
        Parameters:
        - bpe_shapes_list: The multinote shape are learned by bpe algorithms.
          If bpe is not performed, `bpe_shapes_list` should be empty
    """
    start_time = time()

    # Event tokens include:
    # - begin-of-score and end-of-score
    # - shape
    # - track and instrument
    # - positions
    # - measure and time_signature
    # - tempo change

    event_shape_tokens = ['N', 'N~'] + bpe_shapes_list

    event_track_tokens = [
        'R'+int2b36str(i) for i in range(paras['max_track_number'])
    ]
    event_position_tokens = (
        []
        if paras['position_method'] == 'event' else
        ['P'+int2b36str(i) for i in range(get_largest_possible_position(paras['nth']))]
    )
    event_measure_time_sig_tokens = [
        f'M{int2b36str(n)}/{int2b36str(d)}' for n, d in SUPPORTED_TIME_SIGNATURES
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    event_tempo_tokens = [
        'T'+int2b36str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]

    # place padding token in front so that they have same index across all vocabulary
    event_tokens = (
        [PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR]
        + event_shape_tokens + event_track_tokens + event_position_tokens + event_measure_time_sig_tokens + event_tempo_tokens
    )

    max_duration = paras['max_duration']
    pitch_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(128)))
    duration_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(1, max_duration+1)))
    velocity_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(paras['max_track_number'])))
    instrument_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(129)))
    position_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(get_largest_possible_position(paras['nth']))))


    token_count_per_piece = []
    max_measured_count = 0 # max measure number in all pieces
    corpus_measure_time_sig_tokens = set()
    corpus_position_tokens = set() # we want to see how sparse measure and position is
    for piece in corpus_iterator:
        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))
        measures_count = 0
        measures_set = set()
        for t in text_list:
            if t[0] == 'M':
                measures_set.add(t)
                measures_count += 1
        max_measured_count = max(max_measured_count, measures_count)

        corpus_measure_time_sig_tokens.update(measures_set)
        if paras['position_method'] == 'event':
            positions = {t[1:] for t in text_list if t[0] == 'P'}
        else:
            positions = {t.split(':')[-1] for t in text_list if t[0] == 'N'}
        corpus_position_tokens.update(positions)
    corpus_measure_time_sig_tokens = list(corpus_measure_time_sig_tokens)
    corpus_position_tokens = list(corpus_position_tokens)

    summary_string = (
        f'Average tokens per piece: {sum(token_count_per_piece) / len(token_count_per_piece)}\n'\
        f'Event vocab size: {len(event_tokens)}\n'\
        f'- Measure_time_sig event count: {len(event_measure_time_sig_tokens)}\n'\
        f'- Corpus Measure_time_sig: {len(corpus_measure_time_sig_tokens)} ({corpus_measure_time_sig_tokens})\n'\
        f'- Position event count: {len(position_tokens)}\n'\
        f'- Corpus position tokens count: {len(corpus_position_tokens)}\n'\
        f'- Max corpus position: {max(b36str2int(cpts) for cpts in corpus_position_tokens)}\n'\
        f'Largest possible duration: {paras["max_duration"] * paras["nth"] // 4}\n'\
        f'Velocity vocab: {velocity_tokens}\n'\
        f'Tempo vocab: {len(event_tempo_tokens)} ({event_tempo_tokens})\n'\
        f'Vocabulary build time: {time()-start_time}'
    )

    # build a dictionary for every token list
    vocabs = Vocabs(
        paras=paras,
        bpe_shapes_list=bpe_shapes_list,
        padding_token=PADDING_TOKEN_STR,
        events=event_tokens,
        pitchs=pitch_tokens,
        durations=duration_tokens,
        velocities=velocity_tokens,
        track_numbers=track_number_tokens,
        instruments=instrument_tokens,
        positions=position_tokens,
        tempos=event_tempo_tokens,
        time_signatures=event_measure_time_sig_tokens
    )

    return vocabs, summary_string


TOKEN_ATTR_INDEX = {
    'evt': 0, 'events': 0,
    'pit': 1, 'pitchs': 1,
    'dur': 2, 'durations': 2,
    'vel': 3, 'velocities': 3,
    'trn': 4, 'track_numbers': 4,
    'ins': 5, 'instruments': 5,
    'pos': 6, 'positions': 6,
    'mea': 7, 'measure_numbers': 7,
    'tmp': 8, 'tempos': 8,
    'tis': 9, 'time_signatures': 9,
}

COMPLETE_ATTR_NAME = [
    'events', 'pitchs', 'durations', 'velocities', 'track_numbers', 'instruments', 'positions',
    'measure_numbers', 'tempos', 'time_signatures'
]

OUTPUT_ATTR_NAME = [
    'events', 'pitchs', 'durations', 'velocities', 'track_numbers', 'instruments', 'positions'
]


def text_list_to_array(text_list: list, vocabs: Vocabs) -> np.ndarray:
    """
        Serialize pieces into numpy array for the model input.
        Each token is processed into an 9-dimension vector:
            event, pitch, duration, velocity, track_number, instrument, position, tempo, measure_number
        If a token doesn't have an attribute, fill the index of PAD (which should be zero)
        If a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    """
    position_method = vocabs.paras['position_method']
    vocabs.events.text2id = vocabs.events.text2id
    # in build_vocabs, padding token is made to be the 0th element
    padding = 0
    x = np.full((len(text_list), len(COMPLETE_ATTR_NAME)), fill_value=padding, dtype=np.int16)

    tracks_count = 0

    cur_position_id = 0
    cur_measure_number = 0 # measure_number starts at 1, 0 is padding
    cur_tempo_id = padding
    cur_time_sig_id = padding
    cur_measure_cursor = 0
    cur_position_cursor = 0
    for i, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue

        if typename == 'B' or typename == 'E':
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[text]

        elif typename == 'R':
            event_text, instrument = text.split(':')
            track_number = event_text[1:]
            tracks_count += 1
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[event_text]
            x[i][TOKEN_ATTR_INDEX['trn']] = vocabs.track_numbers.text2id[track_number]
            x[i][TOKEN_ATTR_INDEX['ins']] = vocabs.instruments.text2id[instrument]

        elif typename == 'M':
            cur_position_id = vocabs.positions.text2id['0']
            cur_time_sig_id = vocabs.time_signatures.text2id[text]
            cur_measure_number += 1
            cur_measure_cursor = i
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[text]
            x[i][TOKEN_ATTR_INDEX['tmp']] = cur_tempo_id
            x[i][TOKEN_ATTR_INDEX['pos']] = cur_position_id
            x[i][TOKEN_ATTR_INDEX['mea']] = cur_measure_number
            x[i][TOKEN_ATTR_INDEX['tis']] = cur_time_sig_id

        elif typename == 'T':
            event_text, *attr = text = text.split(':')
            cur_tempo_id = vocabs.tempos.text2id[event_text]
            if position_method == 'attribute':
                cur_position_id = b36str2int(attr[0])
            else:
                x[cur_position_cursor][TOKEN_ATTR_INDEX['tmp']] = cur_tempo_id
            # because tempo come after position and/or measure token, which are assigned to previous tempo's id
            x[cur_measure_cursor][TOKEN_ATTR_INDEX['tmp']] = cur_tempo_id
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[event_text]
            x[i][TOKEN_ATTR_INDEX['tmp']] = cur_tempo_id
            x[i][TOKEN_ATTR_INDEX['pos']] = cur_position_id
            x[i][TOKEN_ATTR_INDEX['mea']] = cur_measure_number

        elif typename == 'P':
            cur_position_id = vocabs.positions.text2id[text[1:]]
            cur_position_cursor = i
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[event_text]
            x[i][TOKEN_ATTR_INDEX['pos']] = cur_position_id
            x[i][TOKEN_ATTR_INDEX['mea']] = cur_measure_number
            x[i][TOKEN_ATTR_INDEX['tmp']] = cur_tempo_id
            x[i][TOKEN_ATTR_INDEX['tis']] = cur_time_sig_id

        elif typename == 'N' or typename == 'S':
            event_text, *attr = text.split(':')
            if position_method == 'attribute':
                cur_position_id = vocabs.positions.text2id[attr[4]]
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[event_text]
            x[i][TOKEN_ATTR_INDEX['pit']] = vocabs.pitchs.text2id[attr[0]]
            x[i][TOKEN_ATTR_INDEX['dur']] = vocabs.durations.text2id[attr[1]]
            x[i][TOKEN_ATTR_INDEX['vel']] = vocabs.velocities.text2id[attr[2]]
            x[i][TOKEN_ATTR_INDEX['trn']] = vocabs.track_numbers.text2id[attr[3]]
            assert b36str2int(attr[3]) < tracks_count, 'Invalid track number'
            x[i][TOKEN_ATTR_INDEX['ins']] = x[b36str2int(attr[3])+1, TOKEN_ATTR_INDEX['ins']]
            x[i][TOKEN_ATTR_INDEX['pos']] = cur_position_id
            x[i][TOKEN_ATTR_INDEX['mea']] = cur_measure_number
            x[i][TOKEN_ATTR_INDEX['tmp']] = cur_tempo_id
            x[i][TOKEN_ATTR_INDEX['tis']] = cur_time_sig_id

        else:
            raise ValueError('unknown typename')
    return x


def array_to_text_list(data, vocabs: Vocabs, is_output=False):
    """
        Inverse of text_list_to_array.
    """
    position_method = vocabs.paras['position_method']
    assert len(data.shape) == 2 and data.shape[0] > 0, f'bad numpy array shape: {data.shape}'
    if is_output:
        if position_method == 'event':
            assert data.shape[1] == len(OUTPUT_ATTR_NAME) - 1, f'bad numpy array shape: {data.shape}'
        else:
            assert data.shape[1] == len(OUTPUT_ATTR_NAME), f'bad numpy array shape: {data.shape}'
    else:
        assert data.shape[1] == len(COMPLETE_ATTR_NAME), f'bad numpy array shape: {data.shape}'


    text_list = []
    for i in range(data.shape[0]):
        # in json, key can only be string
        event_text = vocabs.events.id2text[data[i][TOKEN_ATTR_INDEX['evt']]]
        typename = event_text[0]

        if typename == 'B' or typename == 'E':
            text_list.append(event_text) # no additional attribute needed

        elif typename == 'R':
            # track token has instrument attribute
            token_text = event_text + ':' + vocabs.instruments.id2text[data[i][TOKEN_ATTR_INDEX['ins']]]
            text_list.append(token_text)

        elif typename == 'M':
            text_list.append(event_text)

        elif typename == 'T':
            text_list.append(event_text)

        elif typename == 'P':
            text_list.append(event_text)

        elif typename == 'N' or typename == 'S':
            # subtract one because there is padding token at the beginning of the vocab
            track_number = data[i][TOKEN_ATTR_INDEX['trn']] - 1
            token_text = (
                event_text + ':'
                + vocabs.pitchs.id2text[data[i][TOKEN_ATTR_INDEX['pit']]] + ':'
                + vocabs.durations.id2text[data[i][TOKEN_ATTR_INDEX['dur']]] + ':'
                + vocabs.velocities.id2text[data[i][TOKEN_ATTR_INDEX['vel']]] + ':'
                + int2b36str(track_number)
            )
            if position_method == 'attribute':
                token_text += ':' + vocabs.positions.id2text[data[i][TOKEN_ATTR_INDEX['pos']]]
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
    debug_str = 'evt  pit  dur  vel  trn  ins  pos  mea  tmp  tis  reconstructed_text\n'
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
