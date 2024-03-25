import io
from typing import Union, List, Tuple

import numpy as np
from numpy.typing import NDArray

from . import tokens
from .tokens import itob36str
from .vocabs import Vocabs


# TEXT LIST <-> NUMPY ARRAYS

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
