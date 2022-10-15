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
PADDING_TOKEN_STR = '<PAD>'
BEGIN_TOKEN_STR = 'BOS'
END_TOKEN_STR = 'EOS'


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
        return BEGIN_TOKEN_STR

    elif type_priority == TYPE_PRIORITY['EndOfScoreToken']:
        return END_TOKEN_STR

    else:
        raise ValueError(f'bad token namedtuple: {token}')

INSTRUMENT_NAMES = ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano', 'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet', 'Celesta', 'Glockenspiel', 'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer', 'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (jazz)', 'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Guitar harmonics', 'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2', 'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice', 'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn', 'Brass Section', 'Synth Brass 1', 'Synth Brass 2', 'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet', 'Piccolo', 'Flute', 'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina', 'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)', 'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)', 'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bag pipe', 'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot', 'Drumset']