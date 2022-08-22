from io import IOBase
from time import time

from tqdm import tqdm
import numpy as np

from .tokens import (
    tokstr2int,
    tokint2str,
    get_largest_possible_position,
    SPECIAL_TOKENS,
    TOKEN_INT2STR_BASE,
    SUPPORTED_TIME_SIGNATURES,
)

def to_paras_file_path(corpus_dir_path: str):
    pass

def to_corpus_file_path(corpus_dir_path: str):
    pass

def to_shape_vocab_file_path(corpus_dir_path: str):
    pass


def make_corpus_paras_yaml(args: dict) -> str:
    lines = [
        f'{k}: {v}'
        for k, v in args.items()
        if v is not None
    ]
    # print(''.join(lines))
    return '\n'.join(lines)


def parse_corpus_paras_yaml(yaml: str) -> dict:
    pairs = yaml.split('\n')
    args_dict = dict()
    for p in pairs:
        q = p.split(': ')
        if len(q) <= 1:
            continue
        k, v = q
        # tempo_quatization is list
        if k == 'tempo_quantization':
            args_dict[k] = list(map(int, v[1:-1].split(', ')))
        elif k == 'position_method':
            args_dict[k] = v
        elif k == 'use_merge_drums':
            args_dict[k] = (v == 'True')
        else:
            args_dict[k] = int(v)
    return args_dict


def get_corpus_paras(paras_file: IOBase):
    paras_file.seek(0)
    yaml_string = paras_file.read()
    corpus_paras = parse_corpus_paras_yaml(yaml_string)
    return corpus_paras, 

def get_corpus_iterator(corpus_file: IOBase):
    CorpusIterator(corpus_file)

# use iterator to handle large file
class CorpusIterator:
    def __init__(self, corpus_file) -> None:
        self.file = corpus_file
        self.length = None

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
            # if line.startswith('BOS'):
            #     body_start = True
            #     continue
            # if body_start:
            #     yield line[:-1] # remove \n at the end
            if len(line) > 1:
                yield line[-1] # remove \n at the end


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
        max_sample_length: str,
        bpe_shapes_list: list):
    """
        Event tokens include:
        - shape
        - positions in measure
        - measure and time_signature
        - tempo change
        - track and instrument
        - begin-of-score and end-of-score
        Parameters:
        - max_sample_length: we calculate the max amount of measures by checking the most amount of measure tokens
          that can appears within a range of max_sample_length
        - bpe_shapes_list: The multinote shape are learned by bpe algorithms.
          If bpe is not performed, `bpe_shapes_list` should be empty
    """

    start_time = time()

    # shape vacob
    shape_tokens = ['N', 'N~'] + bpe_shapes_list

    # other tokens vocabs
    track_tokens = [
        'R'+tokint2str(i) for i in range(paras['max_track_number'])
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    # tempo_max is inclusive
    tempo_attribute_tokens = list(map(tokint2str, range(tempo_min, tempo_max+tempo_step, tempo_step)))
    position_tokens = [
        'P'+tokint2str(i) for i in range(get_largest_possible_position(paras['nth']))
    ]
    measure_tokens = [
        f'M{tokint2str(n)}/{tokint2str(d)}' for n, d in SUPPORTED_TIME_SIGNATURES
    ]
    if paras['tempo_method'] == 'position_attribute':
        position_tokens = [
            position_token+'+'+tempo_attribute
            for position_token in position_tokens
            for tempo_attribute in tempo_attribute_tokens
        ]

    # place special token in front so that they have same index across all vocabulary
    event_tokens = SPECIAL_TOKENS + ['BOS', 'EOS'] + shape_tokens + track_tokens + measure_tokens + position_tokens

    if paras['tempo_method'] == 'position_event':
        tempo_event_tokens = ['T'+tokint2str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)]
        event_tokens += tempo_event_tokens

    max_duration = paras['max_duration'] * (paras['nth'] // 4)
    pitch_tokens = SPECIAL_TOKENS + list(map(tokint2str, range(128)))
    duration_tokens = SPECIAL_TOKENS + list(map(tokint2str, range(1, max_duration+1))) + [tokint2str(-max_duration)]
    velocity_tokens = SPECIAL_TOKENS + list(map(tokint2str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = SPECIAL_TOKENS + list(map(tokint2str, range(paras['max_track_number'])))
    instrument_tokens = SPECIAL_TOKENS + list(map(tokint2str, range(129)))
    tempo_attribute_tokens = SPECIAL_TOKENS + tempo_attribute_tokens

    token_count_per_piece = []
    # max measure number in all pieces
    max_measure_length = 0
    # the most number of measures span that can fit in the size of max_sample_length
    max_fitting_measure_number = 0
    corpus_measure_tokens = set()
    corpus_position_tokens = set()
    for i, piece in enumerate(piece_iterator):
        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))
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
        f'Average tokens per piece: {sum(token_count_per_piece) / len(token_count_per_piece)}\n'
        + f'Event tokens size: {len(event_tokens)}\n'
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

    return vocab_dicts, summary_string


def text_list_to_numpy(text_list: list, vocabs: dict) -> np.ndarray:
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
            # because tempo come after position, the first position of tempo change will use previous tempo's id 
            x[i-1][6] = cur_tempo_id

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
                x[tokstr2int(attr[3])+1, 5],
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
