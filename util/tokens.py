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

def int2b36str(x: int) -> str:
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

def b36str2int(x: str):
    return int(x, TOKEN_INT2STR_BASE)


# pad token have to be the "zero-th" token
# SPECIAL_TOKENS = ['<PAD>', '<UNK>']
# unknown token may be unessaccery as the whole corpus is preprocessed such that no token would be unknown
SPECIAL_TOKENS = ['<PAD>']


def token_to_str(token: namedtuple) -> str:
    # typename = token.__class__.__name__
    type_priority = token.type_priority

    if type_priority == TYPE_PRIORITY['NoteToken']:
        # event:pitch:duration:velocity:track:instrument(:position)
        # negtive duration means is_cont==True
        text = 'N' if token.duration > 0 else 'N~'
        text += ( f':{int2b36str(token.pitch)}'
                + f':{int2b36str(abs(token.duration))}'
                + f':{int2b36str(token.velocity)}'
                + f':{int2b36str(token.track_number)}')
        if token.position != -1:
            text += f':{int2b36str(token.position)}'
        return text

    elif type_priority == TYPE_PRIORITY['PositionToken']:
        return f'P{int2b36str(token.position)}'

    elif type_priority == TYPE_PRIORITY['MeasureToken']:
        return f'M{int2b36str(token.time_signature[0])}/{int2b36str(token.time_signature[1])}'

    elif type_priority == TYPE_PRIORITY['TempoToken']:
        if token.position == -1:
            return f'T{int2b36str(token.bpm)}'
        return f'T{int2b36str(token.bpm)}:{int2b36str(token.position)}'

    elif type_priority == TYPE_PRIORITY['TrackToken']:
        # type-track:instrument
        return f'R{int2b36str(token.track_number)}:{int2b36str(token.instrument)}'

    elif type_priority == TYPE_PRIORITY['BeginOfScoreToken']:
        return 'BOS'

    elif type_priority == TYPE_PRIORITY['EndOfScoreToken']:
        return 'EOS'

    else:
        raise ValueError(f'bad token namedtuple: {token}')
