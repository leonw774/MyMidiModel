"""
    Define tokens as collections.namedtuple

    For all tokens:
    The time unit of onset is absolute timing. It would be tick at the first stage of preprocessing
    and n-th note at the end.

    For position tokens and measure-related tokens (Measure, Tempo, TimeSignature):
    Except for onset, position, relative_onset, bpm and time_signature, the other attributes are meaningless and should
    always be the default value (-1). The onset time of a position token or a measure-related
    token is the start time of its position or measure in the time unit.

    For track tokens:
    Except for type_priority, track_number and instrument, other attributes are meaningless.

    For begin-of-score and end-of-score token:
    Every attributes should be left as default.

"""

from collections import namedtuple

COMMON_FIELD_NAMES = ['onset', 'type_priority']

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
    COMMON_FIELD_NAMES,
    defaults=[-1, 2]
)
TimeSignatureToken = namedtuple(
    'TimeSignatureToken',
    COMMON_FIELD_NAMES+['time_signature'],
    defaults=[-1, 3, -1]
)
TempoToken = namedtuple(
    'TempoToken',
    COMMON_FIELD_NAMES+['bpm'],
    defaults=[-1, 4, -1]
)
PositionToken = namedtuple(
    'PositionToken',
    COMMON_FIELD_NAMES+['position'],
    defaults=[-1, 5, -1]
)
NoteToken = namedtuple(
    'NoteToken',
    COMMON_FIELD_NAMES+['pitch', 'track_number', 'duration', 'velocity'],
    defaults=[-1, 6, -1, -1, -1, -1]
)
EndOfScoreToken = namedtuple(
    'EndOfScoreToken',
    COMMON_FIELD_NAMES,
    defaults=[float('inf'), 7]
)

SUPPORTED_TIME_SIGNATURES = {
    (numerator, denominator)
    for denominator in [2, 4, 8, 16, 32]
    for numerator in range(1, denominator*3+1)
}

def is_supported_time_signature(numerator, denominator):
    return (numerator, denominator) in SUPPORTED_TIME_SIGNATURES

# python support up to base=36 in int function
ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
TOKEN_INT2STR_BASE = 36

def int2str(x: int) -> str:
    if x == 0:
        return '0'
    isneg = x < 0
    if isneg:
        x = -x
    b = ''
    while x:
        x, d = divmod(x, TOKEN_INT2STR_BASE)
        b = ALPHABET[d] + b
    if isneg:
        b = '-' + b
    return b

def tokenstr2int(x: str):
    return int(x, TOKEN_INT2STR_BASE)

SPECIAL_TOKEN_TEXTS = ['BOS', 'EOS', 'PAD', 'UNK', 'EOP']

class TokenParseError(Exception):
    pass

def token_2_text(token: namedtuple) -> str:
    typename = token.__class__.__name__
    if typename == 'BeginOfScoreToken': # BOS
        return 'BOS'
    elif typename == 'TrackToken': # Track
        # type track:instrument
        return f'R{int2str(token.track_number)}:{int2str(token.instrument)}'
    elif typename == 'MeasureToken': # Measure
        return 'M'
    elif typename == 'TimeSignatureToken': # TimeSig
        return f'S{int2str(token.time_signature[0])}/{int2str(token.time_signature[1])}'
    elif typename == 'TempoToken': # Tempo
        return f'T{int2str(token.bpm)}'
    elif typename == 'PositionToken': # Position
        return f'P{int2str(token.position)}'
    elif typename == 'NoteToken': # Note
        # type pitch:duration:velocity:track:instrument
        return f'N{int2str(token.pitch)}:{int2str(token.duration)}:{int2str(token.velocity)}:{int2str(token.track_number)}'
    elif typename == 'EndOfScoreToken': # EOS
        return 'EOS'
    else:
        raise TokenParseError(f'bad token class name: {typename}')


def text_2_token(text: str):
    typename = text[0]
    if typename == 'B' or typename == 'E' or typename == 'M' :
        attr = tuple()
    elif typename == 'R':
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[1:].split(':'))
    elif typename == 'S':
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[1:].split('/'))
    elif typename == 'T':
        attr = int(text[1:], TOKEN_INT2STR_BASE)
    elif typename == 'P':
        attr = int(text[1:], TOKEN_INT2STR_BASE)
    elif typename == 'N':
        # add note to instrument
        attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[1:].split(':'))
    else:
        raise TokenParseError(f'bad token string: {text}')
    return typename, attr
