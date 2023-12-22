import io
import json
import os
from typing import Union, List, Tuple

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.figure import Figure
import yaml

from . import tokens
from .tokens import b36strtoi, itob36str
from .vocabs import Vocabs

def to_paras_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'paras')

def to_corpus_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'corpus')

def to_pathlist_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'pathlist')

def to_shape_vocab_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'shape_vocab')

def to_vocabs_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'vocabs.json')

def to_arrays_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'arrays.npz')

def dump_corpus_paras(paras_dict: dict) -> str:
    return yaml.dump(paras_dict)

def get_corpus_paras(corpus_dir_path: str) -> dict:
    paras_path = to_paras_file_path(corpus_dir_path)
    with open(paras_path, 'r', encoding='utf8') as paras_file:
        corpus_paras_dict = yaml.safe_load(paras_file)
    return corpus_paras_dict

def get_corpus_vocabs(corpus_dir_path: str) -> Vocabs:
    vocab_path = to_vocabs_file_path(corpus_dir_path)
    with open(vocab_path, 'r', encoding='utf8') as vocabs_file:
        corpus_vocabs_dict = json.load(vocabs_file)
    corpus_vocabs = Vocabs.from_dict(corpus_vocabs_dict)
    return corpus_vocabs


ATTR_NAME_INDEX = {
    # short   long
    'evt': 0, 'events': 0,
    'pit': 1, 'pitchs': 1,
    'dur': 2, 'durations': 2,
    'vel': 3, 'velocities': 3,
    'trn': 4, 'track_numbers': 4,
    'mps': 5, 'mps_numbers': 5,
}

ALL_ATTR_NAMES = [
    'events', 'pitchs', 'durations', 'velocities', 'track_numbers',
    'mps_numbers'
]

OUTPUT_ATTR_NAMES = [
    'events', 'pitchs', 'durations', 'velocities', 'track_numbers'
]


def text_list_to_array(
        text_list: List[str],
        vocabs: Vocabs,
        input_memory: Union[dict, None] = None,
        output_memory: bool = False
    ) -> Union[NDArray[np.uint16], Tuple[NDArray[np.uint16], dict]]:
    """
    Serialize pieces into numpy array for the model input.

    Each token is processed into an 6-dimensional vector:
        event, pitch, duration, velocity, track_number, mps_number

    Positional embedding of head section:
    ```
        B = Begain_of_sequence, T = Track, S = Separator
                sequence: B T T T T T T S ...
              mps number: 1 2 2 2 2 2 2 3 ...
            track number: 0 1 2 3 4 5 6 0 ...
    ```

    Positional embedding of body section:
    ```
        M = Measure, P = Position, N = Multi_note
                sequence: ... M P N N N P N N P N  M  P  N  N  ...
              mps number: ... 4 5 6 6 6 7 8 8 9 10 11 12 13 13 ...
            track number: ... 0 0 1 3 5 0 1 2 0 2  0  0  2  3  ...
    ```
    Basically, head section is treated as a special measure

    If a token has no such attribute, fill it with the index of PAD
    (which should be zero).
    If a token has such attrbute, but it is not in the vocabs,
    exception would raise.
    """
    # in build_vocabs, padding token is made to be the 0th element
    padding = 0
    evt_index = ATTR_NAME_INDEX['evt']
    pit_index = ATTR_NAME_INDEX['pit']
    dur_index = ATTR_NAME_INDEX['dur']
    vel_index = ATTR_NAME_INDEX['vel']
    trn_index = ATTR_NAME_INDEX['trn']
    mps_index = ATTR_NAME_INDEX['mps']

    if input_memory is None:
        last_array_len = 0
        x = np.full(
            (len(text_list), len(ALL_ATTR_NAMES)),
            fill_value=padding,
            dtype=np.uint16
        )
        used_tracks = set()
        cur_mps_number = 0
    elif isinstance(input_memory, dict):
        last_array_len = input_memory['last_array'].shape[0]
        x = np.full(
            (last_array_len+len(text_list), len(ALL_ATTR_NAMES)),
            fill_value=padding,
            dtype=np.uint16
        )
        x[:last_array_len] = input_memory['last_array']
        used_tracks = input_memory['used_tracks']
        cur_mps_number = input_memory['cur_mps_number']
    else:
        raise ValueError('input_memory is not None nor a dictionary')

    for h, text in enumerate(text_list):
        if len(text) > 0:
            typename = text[0]
        else:
            continue
        i = h + last_array_len

        if text in tokens.SPECIAL_TOKENS_STR:
            cur_mps_number += 1
            x[i][evt_index] = vocabs.events.text2id[text]
            x[i][mps_index] = cur_mps_number
            if text == tokens.BEGIN_TOKEN_STR:
                cur_mps_number += 1

        elif typename == tokens.TRACK_EVENTS_CHAR:
            event_text, track_number = text.split(':')
            # instrument = event_text[1:]
            assert track_number not in used_tracks, \
                'Repeating track number in head'
            used_tracks.add(track_number)
            x[i][evt_index] = vocabs.events.text2id[event_text]
            x[i][trn_index] = vocabs.track_numbers.text2id[track_number]
            x[i][mps_index] = cur_mps_number

        elif typename == tokens.MEASURE_EVENTS_CHAR:
            cur_mps_number += 1
            x[i][evt_index] = vocabs.events.text2id[text]
            x[i][mps_index] = cur_mps_number

        elif typename == tokens.TEMPO_EVENTS_CHAR:
            event_text, *attr = text = text.split(':')
            cur_mps_number += 1
            x[i][evt_index] = vocabs.events.text2id[event_text]
            x[i][mps_index] = cur_mps_number

        elif typename == tokens.POSITION_EVENTS_CHAR:
            cur_mps_number += 1
            # cur_position_cursor = i
            x[i][evt_index] = vocabs.events.text2id[text]
            x[i][mps_index] = cur_mps_number

        elif (typename == tokens.NOTE_EVENTS_CHAR
              or typename == tokens.MULTI_NOTE_EVENTS_CHAR):
            event_text, *attr = text.split(':')
            assert attr[3] in used_tracks, 'Invalid track number'
            x[i][evt_index] = vocabs.events.text2id[event_text]
            x[i][pit_index] = vocabs.pitchs.text2id[attr[0]]
            x[i][dur_index] = vocabs.durations.text2id[attr[1]]
            x[i][vel_index] = vocabs.velocities.text2id[attr[2]]
            x[i][trn_index] = vocabs.track_numbers.text2id[attr[3]]
            if x[i-1][trn_index] == 0: # if prev is not note/multi-note
                cur_mps_number += 1
            x[i][mps_index] = cur_mps_number

        else:
            raise ValueError('unknown typename: ' + typename)
    if output_memory:
        return x, {
                'last_array': x,
                'used_tracks': used_tracks,
                # 'cur_position_cursor': cur_position_cursor,
                'cur_mps_number': cur_mps_number,
            }
    else:
        return x


def array_to_text_list(array, vocabs: Vocabs) -> List[str]:
    """
        Inverse of text_list_to_array.
        Expect array to be numpy ndarray or pytorch tensor.
    """
    assert len(array.shape) == 2, f'Bad numpy array shape: {array.shape}'
    assert (array.shape[1] == len(OUTPUT_ATTR_NAMES)
            or array.shape[1] == len(ALL_ATTR_NAMES)), \
            f'Bad numpy array shape: {array.shape}'

    text_list = []
    for x in array:
        # in json, key can only be string
        event_text = vocabs.events.id2text[x[ATTR_NAME_INDEX['evt']]]
        typename = event_text[0]

        if event_text in tokens.SPECIAL_TOKENS_STR:
            if event_text == tokens.PADDING_TOKEN_STR:
                continue
            else:
                text_list.append(event_text) # no additional attribute needed

        elif typename == tokens.TRACK_EVENTS_CHAR:
            # track token has instrument attribute
            token_text = (
                event_text
                + ':'
                + vocabs.track_numbers.id2text[x[ATTR_NAME_INDEX['trn']]]
            )
            text_list.append(token_text)

        elif typename == tokens.MEASURE_EVENTS_CHAR:
            text_list.append(event_text)

        elif typename == tokens.TEMPO_EVENTS_CHAR:
            text_list.append(event_text)

        elif typename == tokens.POSITION_EVENTS_CHAR:
            text_list.append(event_text)

        elif (typename == tokens.NOTE_EVENTS_CHAR
              or typename == tokens.MULTI_NOTE_EVENTS_CHAR):
            # subtract one because there is padding token at index 0
            track_number = x[ATTR_NAME_INDEX['trn']] - 1
            pit_text = vocabs.pitchs.id2text[x[ATTR_NAME_INDEX['pit']]]
            dur_text = vocabs.durations.id2text[x[ATTR_NAME_INDEX['dur']]]
            vel_text = vocabs.velocities.id2text[x[ATTR_NAME_INDEX['vel']]]
            token_text = (
                event_text + ':'
                + pit_text + ':'
                + dur_text + ':'
                + vel_text + ':'
                + itob36str(track_number)
            )
            text_list.append(token_text)
        elif event_text == tokens.PADDING_TOKEN_STR:
            pass
        else:
            raise ValueError(f'unknown typename of event_text: {event_text}')
    # print(track_number_to_event)
    return text_list


def get_full_array_string(input_array: np.ndarray, vocabs: Vocabs):
    array_text_byteio = io.BytesIO()
    np.savetxt(array_text_byteio, input_array, fmt='%d')
    array_savetxt_list = array_text_byteio.getvalue().decode().split('\n')
    reconstructed_text_list = array_to_text_list(input_array, vocabs=vocabs)
    short_attr_names = [
        attr_name
        for attr_name in ATTR_NAME_INDEX
        if len(attr_name) == 3
    ]
    sorted_attr_names = sorted(
        short_attr_names, key=ATTR_NAME_INDEX.get)
    debug_str_head = ' '.join([
        f'{attr_name:>4}'
        for attr_name in sorted_attr_names
    ])
    debug_str_head += '  reconstructed_text\n'
    debug_str = debug_str_head + (
        '\n'.join([
            ' '.join([f'{s:>4}' for s in a.split()])
            + f'  {z}'
            for a, z in zip(array_savetxt_list, reconstructed_text_list)
        ])
    )
    return debug_str


PIANOROLL_MEASURELINE_COLOR = (0.8, 0.8, 0.8, 1.0)
PIANOROLL_TRACK_COLORMAP = cm.get_cmap('terrain')
PIANOROLL_SHAPE_LINE_COLOR = (0.88, 0.2, 0.24, 0.8)

def piece_to_roll(piece: str, tpq: int) -> Figure:
    plt.clf()
    text_list = piece.split(' ')

    cur_measure_length = 0
    cur_measure_onset = 0
    # know how long this piece is and draw measure lines
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == tokens.MEASURE_EVENTS_CHAR:
            numer, denom = (b36strtoi(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_measure_length = 4 * tpq * numer // denom

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
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == tokens.TRACK_EVENTS_CHAR:
            total_track_number += 1
            instrument, track_number = (
                b36strtoi(x)
                for x in text[1:].split(':')
            )
            if instrument == 128:
                drum_tracks.add(track_number)

        elif typename == tokens.MEASURE_EVENTS_CHAR:
            if is_head:
                track_colors = [
                    PIANOROLL_TRACK_COLORMAP(i / total_track_number, alpha=0.5)
                    for i in range(total_track_number)
                ]
                is_head = False
            numer, denom = (b36strtoi(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset
            cur_measure_length = 4 * tpq * numer // denom
            # if cur_time_signature != (numer, denom):
            #     cur_time_signature = (numer, denom)
            # draw measure line between itself and the next measure
            plt.axvline(
                x=cur_time,
                ymin=0,
                ymax=128,
                color=PIANOROLL_MEASURELINE_COLOR,
                linewidth=0.5,
                zorder=0
            )

        elif typename == tokens.POSITION_EVENTS_CHAR:
            position = b36strtoi(text[1:])
            cur_time = position + cur_measure_onset

        elif typename == tokens.TEMPO_EVENTS_CHAR:
            if ':' in text[1:]:
                tempo, position = (b36strtoi(x) for x in text[1:].split(':'))
                cur_time = position + cur_measure_onset
            else:
                tempo = b36strtoi(text[1:])
            plt.annotate(
                xy=(cur_time+0.05, 0.95),
                text=f'bpm={tempo}',
                xycoords=('data', 'axes fraction')
            )

        elif typename == tokens.NOTE_EVENTS_CHAR:
            is_cont = text[1] == '~'
            if is_cont:
                note_attr = tuple(b36strtoi(x) for x in text[3:].split(':'))
            else:
                note_attr = tuple(b36strtoi(x) for x in text[2:].split(':'))
            if len(note_attr) == 5:
                cur_time = note_attr[4] + cur_measure_onset
                note_attr = note_attr[:4]
            pitch, duration, _, track_number = note_attr
            if track_number in drum_tracks:
                current_axis.add_patch(
                    Ellipse(
                        xy=(onset_time+0.5, pitch+0.5),
                        width=max_time / 256 * 0.25,
                        height=0.5,
                        edgecolor='grey',
                        linewidth=0.75,
                        facecolor=track_colors[track_number],
                        fill=True
                    )
                )
            else:
                current_axis.add_patch(
                    Rectangle(
                        xy=(cur_time, pitch),
                        width=duration-0.25,
                        height=0.75,
                        edgecolor='grey',
                        linewidth=0.75,
                        facecolor=track_colors[track_number],
                        fill=True
                    )
                )

        elif typename == tokens.MULTI_NOTE_EVENTS_CHAR:
            shape_string, *other_attrs = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                if s[-1] == '~':
                    relnote = [b36strtoi(a) for a in s[:-1].split(',')]
                else:
                    relnote = [b36strtoi(a) for a in s.split(',')]
                relnote = [is_cont] + relnote
                relnote_list.append(relnote)
            base_pitch, stretch_factor, _velocity, track_number = map(
                b36strtoi,
                other_attrs
            )
            if len(position) == 1:
                cur_time = position[0] + cur_measure_onset
            prev_pitch = -1
            prev_onset_time = -1
            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                pitch = base_pitch + rel_pitch
                duration = rel_dur * stretch_factor
                onset_time = cur_time + rel_onset * stretch_factor
                if track_number in drum_tracks:
                    current_axis.add_patch(
                        Ellipse(
                            xy=(onset_time+0.5, pitch+0.5),
                            width=max_time / 256 * 0.25,
                            height=0.5,
                            edgecolor='grey',
                            linewidth=0.75,
                            facecolor=track_colors[track_number],
                            fill=True

                        )
                    )
                else:
                    current_axis.add_patch(
                        Rectangle(
                            xy=(cur_time, pitch),
                            width=duration-0.25,
                            height=0.75,
                            edgecolor='grey',
                            linewidth=0.75,
                            facecolor=track_colors[track_number],
                            fill=True
                        )
                    )
                if prev_pitch != -1:
                    plt.plot(
                        scalex=[prev_onset_time+0.4, onset_time+0.4],
                        scaley=[prev_pitch+0.4, pitch+0.4],
                        color=PIANOROLL_SHAPE_LINE_COLOR,
                        linewidth=1.0,
                        markerfacecolor=track_colors[track_number],
                        marker='.',
                        markersize=1
                    )
                prev_pitch = pitch
                prev_onset_time = onset_time
        else:
            if text not in tokens.SPECIAL_TOKENS_STR:
                raise ValueError(f'bad token string: {text}')
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.xlim(xmin=0)
    plt.autoscale()
    plt.margins(x=0)
    plt.tight_layout()
    return plt.gcf()
