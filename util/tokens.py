from collections import namedtuple

_COMMON_FIELD_NAMES = ['onset', 'type_priority']
TYPE_PRIORITY = {
    'BeginOfScoreToken': 0,
    'TrackToken': 1,
    'SectionSeperatorToken': 2,
    'MeasureToken': 3,
    'PositionToken': 4,
    'TempoToken': 5,
    'NoteToken': 6,
    'EndOfScoreToken': 7,
}

BeginOfScoreToken = namedtuple(
    'BeginOfScoreToken',
    _COMMON_FIELD_NAMES,
    defaults=[-1, TYPE_PRIORITY['BeginOfScoreToken']]
)
TrackToken = namedtuple(
    'TrackToken',
    _COMMON_FIELD_NAMES + ['track_number', 'instrument'],
    defaults=[-1, TYPE_PRIORITY['TrackToken'], -1, -1]
)
SectionSeperatorToken = namedtuple(
    'SectionSeperatorToken',
    _COMMON_FIELD_NAMES,
    defaults=[-1, TYPE_PRIORITY['SectionSeperatorToken']]
)
MeasureToken = namedtuple(
    'MeasureToken',
    _COMMON_FIELD_NAMES + ['time_signature'],
    defaults=[-1, TYPE_PRIORITY['MeasureToken'], -1]
)
PositionToken = namedtuple(
    'PositionToken',
    _COMMON_FIELD_NAMES + ['position'],
    defaults=[-1, TYPE_PRIORITY['PositionToken'], -1]
)
TempoToken = namedtuple(
    'TempoToken',
    _COMMON_FIELD_NAMES + ['bpm', 'position'],
    defaults=[-1, TYPE_PRIORITY['TempoToken'], -1, -1]
)
NoteToken = namedtuple(
    'NoteToken',
    _COMMON_FIELD_NAMES + [
        'track_number', 'pitch', 'duration', 'velocity', 'position'
    ],
    defaults=[-1, TYPE_PRIORITY['NoteToken'], -1, -1, -1, -1, -1]
)
EndOfScoreToken = namedtuple(
    'EndOfScoreToken',
    _COMMON_FIELD_NAMES,
    defaults=[float('inf'), TYPE_PRIORITY['EndOfScoreToken'],]
)

SUPPORT_TIME_SIGNATURE_N_MAX = 24
SUPPORT_TIME_SIGNATURE_D_LOG2_MAX = 3
SUPPORT_TIME_SIGNATURE_ND_RATIO = 3.0

# determined with parameters to cover almost all reasonable usages
def get_supported_time_signatures(
        numerator_max: int = SUPPORT_TIME_SIGNATURE_N_MAX,
        denominator_log2_max: int = SUPPORT_TIME_SIGNATURE_D_LOG2_MAX,
        nd_raio_max: float = SUPPORT_TIME_SIGNATURE_ND_RATIO) -> set:
    return {
        (numerator, denominator)
        for denominator in [
            int(2 ** (log2d+1)) for log2d in range(denominator_log2_max)]
        for numerator in range(
            1, min(numerator_max, int(denominator * nd_raio_max)) + 1)
    }

# determined with hand-picked items based on occurence in datasets
# def get_suported_time_signatures():
#     supported_time_signatures = {
#         (2,2),
#         (1,4), (2,4), (3,4), (4,4), (5,4),
#         (3,8), (6,8),
#     }
#     return supported_time_signatures


def get_largest_possible_position(
        tpq: int,
        supported_time_signatures: set = None) -> int:
    if supported_time_signatures is None:
        return max(
            4 * tpq * s[0] // s[1]
            for s in get_supported_time_signatures() # use default
        )
    return max(
        4 * tpq * s[0] // s[1]
        for s in supported_time_signatures
    )

# python support up to base=36 in int function
TOKEN_INT2STR_BASE = 36

def itob36str(x: int) -> str:
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

def b36strtoi(x: str):
    return int(x, TOKEN_INT2STR_BASE)


# pad token have to be the first token (a.k.a. index 0) in any vocabulary
PADDING_TOKEN_STR   = 'PAD'
BEGIN_TOKEN_STR     = 'BOS'
END_TOKEN_STR       = 'EOS'
SEP_TOKEN_STR       = 'SEP'
SPECIAL_TOKENS_STR = [
    PADDING_TOKEN_STR,
    BEGIN_TOKEN_STR,
    END_TOKEN_STR,
    SEP_TOKEN_STR
]

TRACK_EVENTS_CHAR       = 'R'
MEASURE_EVENTS_CHAR     = 'M'
POSITION_EVENTS_CHAR    = 'P'
TEMPO_EVENTS_CHAR       = 'T'
NOTE_EVENTS_CHAR        = 'N'
MULTI_NOTE_EVENTS_CHAR  = 'U'

def token_to_str(token: namedtuple) -> str:
    # typename = token.__class__.__name__
    type_priority = token.type_priority

    if type_priority == TYPE_PRIORITY['NoteToken']:
        # event:pitch:duration:velocity:track:instrument(:position)
        # negative token.duration means is_cont==True
        text = NOTE_EVENTS_CHAR
        if token.duration > 0:
            text += '~'
        text += ( f':{itob36str(token.pitch)}'
                + f':{itob36str(abs(token.duration))}'
                + f':{itob36str(token.velocity)}'
                + f':{itob36str(token.track_number)}')
        if token.position != -1:
            text += f':{itob36str(token.position)}'
        return text

    elif type_priority == TYPE_PRIORITY['PositionToken']:
        return f'{POSITION_EVENTS_CHAR}{itob36str(token.position)}'

    elif type_priority == TYPE_PRIORITY['MeasureToken']:
        return (
            f'{MEASURE_EVENTS_CHAR}'
            f'{itob36str(token.time_signature[0])}'
            f'/'
            f'{itob36str(token.time_signature[1])}'
        )

    elif type_priority == TYPE_PRIORITY['TempoToken']:
        if token.position == -1:
            return f'{TEMPO_EVENTS_CHAR}{itob36str(token.bpm)}'
        return (
            f'{TEMPO_EVENTS_CHAR}{itob36str(token.bpm)}'
            f':'
            f'{itob36str(token.position)}'
        )

    elif type_priority == TYPE_PRIORITY['TrackToken']:
        return (
            f'{TRACK_EVENTS_CHAR}'
            f'{itob36str(token.instrument)}'
            f':{itob36str(token.track_number)}'
        )

    elif type_priority == TYPE_PRIORITY['BeginOfScoreToken']:
        return BEGIN_TOKEN_STR

    elif type_priority == TYPE_PRIORITY['SectionSeperatorToken']:
        return SEP_TOKEN_STR

    elif type_priority == TYPE_PRIORITY['EndOfScoreToken']:
        return END_TOKEN_STR

    else:
        raise ValueError(f'bad token namedtuple: {token}')

INSTRUMENT_NAMES = [
    'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano',
    'Honky-tonk Piano', 'Electric Piano 1', 'Electric Piano 2',
    'Harpsichord', 'Clavinet', 'Celesta', 'Glockenspiel',
    'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer',
    'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ',
    'Accordion', 'Harmonica', 'Tango Accordion',
    'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)',
    'Electric Guitar (jazz)', 'Electric Guitar (clean)', 'Electric Guitar (muted)',
    'Overdriven Guitar', 'Distortion Guitar', 'Guitar harmonics',
    'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass',
    'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2',
    'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings',
    'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2',
    'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice',
    'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn',
    'Brass Section', 'Synth Brass 1', 'Synth Brass 2',
    'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax',
    'Oboe', 'English Horn', 'Bassoon', 'Clarinet', 'Piccolo', 'Flute',
    'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina',
    'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)',
    'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)',
    'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)',
    'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)',
    'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)',
    'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)',
    'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bag pipe', 'Fiddle', 'Shanai',
    'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom',
    'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore',
    'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot', 'Drumset'
]
