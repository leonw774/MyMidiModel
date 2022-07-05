"""
    Define tokens as collections.namedtuple

    For all tokens:
    The time unit of onset is absolute timing. The unit is n-th note.

    For position tokens and measure-related tokens (Measure, Tempo):
    Except for onset, position, bpm and time_signature, the other attributes are meaningless and should
    always be the default value (-1). The onset time of a position token or a measure-related
    token is the start time of its position or measure in the time unit.

    For track tokens:
    Except for type_priority, track_number and instrument, other attributes are meaningless.

    For begin-of-score and end-of-score token:
    Every attributes should be left as default.
"""

import re
from collections import namedtuple

COMMON_FIELD_NAMES = ['onset', 'type_priority']
TYPE_PRIORITY = {
    'BeginOfScoreToken' : 0,
    'TrackToken' : 1,
    'MeasureToken' : 2,
    'TempoToken' : 3,
    'PositionToken' : 4,
    'NoteToken' : 5,
    'EndOfScoreToken' : 6,
}

BeginOfScoreToken = namedtuple(
    'BeginOfScoreToken',
    COMMON_FIELD_NAMES,
    defaults=[-1, 0]
)
TrackToken = namedtuple(
    'TrackToken',
    COMMON_FIELD_NAMES+['track_number', 'instrument'],
    defaults=[-1, 1, -1, -1]
)
MeasureToken = namedtuple(
    'MeasureToken',
    COMMON_FIELD_NAMES+['time_signature', 'bpm'],
    defaults=[-1, 2, -1, -1]
)
TempoToken = namedtuple(
    'TempoToken',
    COMMON_FIELD_NAMES+['bpm'],
    defaults=[-1, 3, -1]
)
PositionToken = namedtuple(
    'PositionToken',
    COMMON_FIELD_NAMES+['position', 'bpm'],
    defaults=[-1, 4, -1, -1]
)
NoteToken = namedtuple(
    'NoteToken',
    COMMON_FIELD_NAMES+['pitch', 'track_number', 'duration', 'velocity'],
    defaults=[-1, 5, -1, -1, -1, -1]
)
EndOfScoreToken = namedtuple(
    'EndOfScoreToken',
    COMMON_FIELD_NAMES,
    defaults=[float('inf'), 6]
)

SUPPORTED_TIME_SIGNATURES = {
    (numerator, denominator)
    for denominator in [2, 4, 8, 16, 32]
    for numerator in range(1, denominator*3+1)
}

def get_largest_possible_position(nth: int) -> int:
    return max(
        s[0] * (nth // s[1])
        for s in SUPPORTED_TIME_SIGNATURES
    )

def is_supported_time_signature(numerator, denominator):
    return (numerator, denominator) in SUPPORTED_TIME_SIGNATURES

# python support up to base=36 in int function
TOKEN_INT2STR_BASE = 36

def tokenint2str(x: int) -> str:
    if 0 <= x < TOKEN_INT2STR_BASE:
        return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[x]
    isneg = x < 0
    if isneg:
        x = -x
    b = ''
    while x:
        x, d = divmod(x, TOKEN_INT2STR_BASE)
        b = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[d] + b
    return ('-' + b) if isneg else b

def tokenstr2int(x: str):
    return int(x, TOKEN_INT2STR_BASE)

SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[UNK]']
# SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[UNK]', '[SEP]', '[START]', '[END]']

class TokenParseError(Exception):
    pass

def token_to_text(token: namedtuple) -> str:
    # typename = token.__class__.__name__
    type_priority = token.type_priority
    if type_priority == 6: #TYPE_PRIORITY['EndOfScoreToken']:
        return 'EOS'
    elif type_priority == 5: #TYPE_PRIORITY['NoteToken']:
        # type-pitch:duration:velocity:track:instrument
        return f'N{tokenint2str(token.pitch)}:{tokenint2str(token.duration)}:{tokenint2str(token.velocity)}:{tokenint2str(token.track_number)}'
    elif type_priority == 4: #TYPE_PRIORITY['PositionToken']:
        if token.bpm == -1:
            return f'P{tokenint2str(token.position)}'
        else:
            return f'P{tokenint2str(token.position)}+{tokenint2str(token.bpm)}'
    elif type_priority == 3: #TYPE_PRIORITY['TempoToken']:
        return f'T{tokenint2str(token.bpm)}'
    elif type_priority == 2: #TYPE_PRIORITY['MeasureToken']:
        if token.bpm == -1:
            return f'M{tokenint2str(token.time_signature[0])}/{tokenint2str(token.time_signature[1])}'
        else:
            return f'M{tokenint2str(token.time_signature[0])}/{tokenint2str(token.time_signature[1])}+{tokenint2str(token.bpm)}'
    elif type_priority == 1: #TYPE_PRIORITY['TrackToken']:
        # type-track:instrument
        return f'R{tokenint2str(token.track_number)}:{tokenint2str(token.instrument)}'
    elif type_priority == TYPE_PRIORITY['BeginOfScoreToken']:
        return 'BOS'
    else:
        raise TokenParseError(f'bad token namedtuple: {token}')


def text_to_attrs(text: str):
    typename = text[0]
    if typename == 'B' or typename == 'E':
        attr = tuple()
    elif typename == 'R':
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[1:].split(':'))
    elif typename == 'M':
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in re.split(r'[:/+]', text[1:]))
    elif typename == 'T':
        attr = (int(text[1:], TOKEN_INT2STR_BASE), )
    elif typename == 'P':
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in re.split(r'[:+]', text[1:]))
    elif typename == 'N':
        # add note to instrument
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[1:].split(':'))
    else:
        raise TokenParseError(f'bad token string: {text}')
    return typename, attr
