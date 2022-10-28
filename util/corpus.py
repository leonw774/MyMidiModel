import io
import json
import os

import numpy as np
import matplotlib
from matplotlib.patches import Rectangle, Ellipse
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import yaml

from .vocabs import Vocabs
from .tokens import b36str2int, int2b36str, PADDING_TOKEN_STR

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
    def __init__(self, corpus_dir_path: str) -> None:
        self.file_path = os.path.join(corpus_dir_path, 'corpus')
        self.length = None
        self.file = None

    # context manager
    def __enter__(self):
        self.file = open(self.file_path, 'r', encoding='utf8')
        return self

    def __exit__(self, _value, _type, _trackback):
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
        Each token is processed into an 9-dimensional vector:
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
            x[i][TOKEN_ATTR_INDEX['evt']] = vocabs.events.text2id[text]
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


PIANOROLL_MEASURELINE_COLOR = (0.8, 0.8, 0.8, 1.0)
PIANOROLL_TRACK_COLORMAP = matplotlib.colormaps['terrain']
PIANOROLL_SHAPE_LINE_COLOR = (0.88, 0.2, 0.24, 0.8)

def piece_to_roll(piece: str, nth: int) -> Figure:
    plt.clf()
    text_list = piece.split(' ')

    cur_measure_length = 0
    cur_measure_onset = 0
    # know how long this piece is and draw measure lines
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == 'M':
            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_measure_length = round(nth * numer / denom)

    max_time = cur_measure_onset+cur_measure_length
    figure_width = min(max_time*0.1, 128)
    plt.figure(figsize=(figure_width, 12.8))
    # plt.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.05)
    current_axis = plt.gca()

    is_head = True
    track_colors = []
    drum_tracks = set()
    total_track_number = 0
    cur_time = 0
    cur_measure_length = 0
    cur_measure_onset = 0
    # cur_time_signature = None
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == 'R':
            total_track_number += 1
            track_number, instrument = (b36str2int(x) for x in text[1:].split(':'))
            if instrument == 128:
                drum_tracks.add(track_number)

        elif typename == 'M':
            if is_head:
                track_colors = [
                    PIANOROLL_TRACK_COLORMAP(i / total_track_number, alpha=0.5)
                    for i in range(total_track_number)
                ]
                is_head = False
            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset
            cur_measure_length = round(nth * numer / denom)
            # if cur_time_signature != (numer, denom):
            #     cur_time_signature = (numer, denom)
            # draw measure line between itself and the next measure
            plt.axvline(x=cur_time, ymin=0, ymax=128, color=PIANOROLL_MEASURELINE_COLOR, linewidth=0.5, zorder=0)

        elif typename == 'P':
            position = b36str2int(text[1:])
            cur_time = position + cur_measure_onset

        elif typename == 'T':
            if ':' in text[1:]:
                tempo, position = (b36str2int(x) for x in text[1:].split(':'))
                cur_time = position + cur_measure_onset
            else:
                tempo = b36str2int(text[1:])
            plt.annotate(xy=(cur_time+0.05, 0.95), text=f'bpm={tempo}', xycoords=('data', 'axes fraction'))

        elif typename == 'N':
            is_cont = (text[1] == '~')
            if is_cont:
                note_attr = tuple(b36str2int(x) for x in text[3:].split(':'))
            else:
                note_attr = tuple(b36str2int(x) for x in text[2:].split(':'))
            if len(note_attr) == 5:
                cur_time = note_attr[4] + cur_measure_onset
                note_attr = note_attr[:4]
            pitch, duration, velocity, track_number = note_attr
            if track_number in drum_tracks:
                current_axis.add_patch(
                    Ellipse(
                        (cur_time+0.4, pitch+0.4), max_time / 256 * 0.1, 0.4,
                        edgecolor='grey', linewidth=0.75, facecolor=track_colors[track_number], fill=True
                    )
                )
            else:
                current_axis.add_patch(
                    Rectangle(
                        (cur_time, pitch), duration-0.2, 0.8,
                        edgecolor='grey', linewidth=0.75, facecolor=track_colors[track_number], fill=True
                    )
                )

        elif typename == 'S':
            shape_string, *other_attr = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                is_cont = (s[-1] == '~')
                if is_cont:
                    relnote = [b36str2int(a) for a in s[:-1].split(',')]
                else:
                    relnote = [b36str2int(a) for a in s.split(',')]
                relnote = [is_cont] + relnote
                relnote_list.append(relnote)
            base_pitch, time_unit, velocity, track_number, *position = (b36str2int(x) for x in other_attr)
            if len(position) == 1:
                cur_time = position[0] + cur_measure_onset
            prev_pitch = None
            prev_onset_time = None
            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                pitch = base_pitch + rel_pitch
                duration = rel_dur * time_unit
                onset_time = cur_time + rel_onset * time_unit
                if track_number in drum_tracks:
                    current_axis.add_patch(
                        Ellipse(
                            (onset_time+0.4, pitch+0.4), max_time / 256 * 0.2, 0.4,
                            edgecolor='grey', linewidth=0.75, facecolor=track_colors[track_number], fill=True
                        )
                    )
                else:
                    current_axis.add_patch(
                        Rectangle(
                            (onset_time, pitch), duration-0.2, 0.8,
                            edgecolor='grey', linewidth=0.75, facecolor=track_colors[track_number], fill=True
                        )
                    )
                if prev_pitch is not None:
                    plt.plot(
                        [prev_onset_time+0.4, onset_time+0.4], [prev_pitch+0.4, pitch+0.4],
                        color=PIANOROLL_SHAPE_LINE_COLOR,  linewidth=1.0,
                        markerfacecolor=track_colors[track_number], marker='.', markersize=1
                    )
                prev_pitch = pitch
                prev_onset_time = onset_time
        else:
            raise ValueError(f'bad token string: {text}')
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.xlim(xmin=0)
    plt.autoscale()
    plt.margins(x=0)
    plt.tight_layout()
    return plt.gcf()


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
