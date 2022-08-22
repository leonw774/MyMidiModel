"""
    Define tokens as collections.namedtuple

    For all tokens:
    The time unit of onset is absolute timing. The unit is n-th note.

    For position tokens, tempo tokens and measure tokens:
    Except for onset, position, bpm and time_signature, the other attributes are meaningless and should
    always be the default value (-1). The onset time of a position token or a measure-related
    token is the start time of its position or measure in the time unit.

    For track tokens:
    Except for type_priority, track_number and instrument, other attributes are meaningless.

    For begin-of-score and end-of-score token:
    Every attributes should be left as default.
"""

from collections import namedtuple

_COMMON_FIELD_NAMES = ['onset', 'type_priority']
TYPE_PRIORITY = {
    'BeginOfScoreToken': 0,
    'TrackToken': 1,
    'MeasureToken': 2,
    'PositionToken': 3,
    'TempoToken': 4,
    'NoteToken': 5,
    'EndOfScoreToken': 6,
}

BeginOfScoreToken = namedtuple(
    'BeginOfScoreToken',
    _COMMON_FIELD_NAMES,
    defaults=[-1, TYPE_PRIORITY['BeginOfScoreToken']]
)
TrackToken = namedtuple(
    'TrackToken',
    _COMMON_FIELD_NAMES+['track_number', 'instrument'],
    defaults=[-1, TYPE_PRIORITY['TrackToken'], -1, -1]
)

MeasureToken = namedtuple(
    'MeasureToken',
    _COMMON_FIELD_NAMES+['time_signature'],
    defaults=[-1, TYPE_PRIORITY['MeasureToken'], -1]
)
PositionToken = namedtuple(
    'PositionToken',
    _COMMON_FIELD_NAMES+['position'],
    defaults=[-1, TYPE_PRIORITY['PositionToken'], -1]
)
TempoToken = namedtuple(
    'TempoToken',
    _COMMON_FIELD_NAMES+['bpm', 'position'],
    defaults=[-1, TYPE_PRIORITY['TempoToken'], -1, -1]
)
NoteToken = namedtuple(
    'NoteToken',
    _COMMON_FIELD_NAMES+['track_number', 'pitch', 'duration', 'velocity', 'position'],
    defaults=[-1, TYPE_PRIORITY['NoteToken'], -1, -1, -1, -1, -1]
)
EndOfScoreToken = namedtuple(
    'EndOfScoreToken',
    _COMMON_FIELD_NAMES,
    defaults=[float('inf'), TYPE_PRIORITY['EndOfScoreToken'],]
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

def tokint2str(x: int) -> str:
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

def tokstr2int(x: str):
    return int(x, TOKEN_INT2STR_BASE)


SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[UNK]']
# SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[UNK]', '[SEP]', '[START]', '[END]']


class TokenParseError(Exception):
    pass


def token_to_text(token: namedtuple) -> str:
    # typename = token.__class__.__name__
    type_priority = token.type_priority

    if type_priority == TYPE_PRIORITY['NoteToken']:
        # event:pitch:duration:velocity:track:instrument(:position)
        text = 'N' if token.duration > 0 else 'N~'
        text += ( f':{tokint2str(token.pitch)}'
                + f':{tokint2str(token.duration)}'
                + f':{tokint2str(token.velocity)}'
                + f':{tokint2str(token.track_number)}')
        if token.position != -1:
            text += f':{tokint2str(token.position)}'
        return text

    elif type_priority == TYPE_PRIORITY['PositionToken']:
        return f'P{tokint2str(token.position)}'

    elif type_priority == TYPE_PRIORITY['MeasureToken']:
        return f'M{tokint2str(token.time_signature[0])}/{tokint2str(token.time_signature[1])}'


    elif type_priority == TYPE_PRIORITY['TempoToken']:
        return f'T{tokint2str(token.bpm)}' if token.position == -1 else f'T{tokint2str(token.bpm)}:{tokint2str(token.position)}'

    elif type_priority == TYPE_PRIORITY['TrackToken']:
        # type-track:instrument
        return f'R{tokint2str(token.track_number)}:{tokint2str(token.instrument)}'

    elif type_priority == TYPE_PRIORITY['BeginOfScoreToken']:
        return 'BOS'

    elif type_priority == TYPE_PRIORITY['EndOfScoreToken']:
        return 'EOS'

    else:
        raise TokenParseError(f'bad token namedtuple: {token}')
