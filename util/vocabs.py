# from util.corpus import CorpusReader # will cause circular import
from .tokens import (
    BEGIN_TOKEN_STR,
    END_TOKEN_STR,
    PADDING_TOKEN_STR,
    int2b36str,
    get_largest_possible_position,
    SUPPORTED_TIME_SIGNATURES
)

class Vocabs:

    class AttrVocab:
        size: int
        id2text: dict
        text2id: dict
        def __init__(self, init_obj) -> None:
            if isinstance(init_obj, list):
                self.size = len(init_obj)
                self.id2text = {i:t for i, t in enumerate(init_obj)}
                self.text2id = {t:i for i, t in enumerate(init_obj)}
            elif isinstance(init_obj, dict):
                self.size = init_obj['size']
                self.id2text = {int(k): v for k, v in init_obj['id2text'].items()} # key are string in json
                self.text2id = init_obj['text2id']
            else:
                raise TypeError(f'AttrVocab\'s init_obj can only be list or dict. Got {init_obj}')
        def __len__(self) -> int:
            return self.size

    paras: dict
    bpe_shapes_list: list
    padding_token: str
    events: AttrVocab
    pitchs: AttrVocab
    durations: AttrVocab
    velocities: AttrVocab
    track_numbers: AttrVocab
    instruments: AttrVocab
    positions: AttrVocab
    tempos: AttrVocab

    def __init__(self, paras: dict, bpe_shapes_list: list, padding_token: str,
            events, pitchs, durations, velocities, track_numbers, instruments, positions, tempos, time_signatures) -> None:
        self.paras = paras
        self.bpe_shapes_list = bpe_shapes_list
        self.padding_token = padding_token
        self.events = Vocabs.AttrVocab(events)
        self.pitchs = Vocabs.AttrVocab(pitchs)
        self.durations = Vocabs.AttrVocab(durations)
        self.velocities = Vocabs.AttrVocab(velocities)
        self.track_numbers = Vocabs.AttrVocab(track_numbers)
        self.instruments = Vocabs.AttrVocab(instruments)
        self.positions = Vocabs.AttrVocab(positions)
        self.tempos = Vocabs.AttrVocab(tempos)
        self.time_signatures = Vocabs.AttrVocab(time_signatures)

    @classmethod
    def from_dict(cls, d):
        return cls(
            d['paras'], d['bpe_shapes_list'], d['padding_token'],
            d['events'], d['pitchs'], d['durations'],
            d['velocities'], d['track_numbers'], d['instruments'],
            d['positions'], d['tempos'], d['time_signatures']
        )

    def to_dict(self) -> dict:
        return {
            'paras': self.paras,
            'bpe_shapes_list': self.bpe_shapes_list,
            'padding_token': self.padding_token,
            'events': vars(self.events),
            'pitchs': vars(self.pitchs),
            'durations': vars(self.durations),
            'velocities': vars(self.velocities),
            'track_numbers': vars(self.track_numbers),
            'instruments': vars(self.instruments),
            'positions': vars(self.positions),
            'tempos': vars(self.tempos),
            'time_signatures': vars(self.time_signatures),
        }


def build_vocabs(
        corpus_reader,
        paras: dict,
        bpe_shapes_list: list):
    """
        Parameters:
        - bpe_shapes_list: The multinote shape are learned by bpe algorithms.
          If bpe is not performed, `bpe_shapes_list` should be empty
    """
    # start_time = time()

    # Event tokens include:
    # - begin-of-score and end-of-score
    # - shape
    # - track and instrument
    # - positions
    # - measure and time_signature
    # - tempo change

    event_shape_tokens = ['N', 'N~'] + bpe_shapes_list

    event_track_tokens = [
        'R'+int2b36str(i) for i in range(paras['max_track_number'])
    ]
    event_position_tokens = (
        ['P'+int2b36str(i) for i in range(get_largest_possible_position(paras['nth']))]
        if paras['position_method'] == 'event' else
        []
    )
    event_measure_time_sig_tokens = [
        f'M{int2b36str(n)}/{int2b36str(d)}' for n, d in SUPPORTED_TIME_SIGNATURES
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    event_tempo_tokens = [
        'T'+int2b36str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]

    # place padding token in front so that they have same index across all vocabulary
    event_tokens = (
        [PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR]
        + event_shape_tokens + event_track_tokens + event_position_tokens + event_measure_time_sig_tokens + event_tempo_tokens
    )

    max_duration = paras['max_duration']
    pitch_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(128)))
    duration_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(1, max_duration+1)))
    velocity_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(paras['max_track_number'])))
    instrument_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(129)))
    position_tokens = [PADDING_TOKEN_STR] + list(map(int2b36str, range(get_largest_possible_position(paras['nth']))))

    token_count_per_piece = []
    # max_measured_count = 0 # find max measure number in all pieces
    # corpus_measure_time_sig_tokens = set()
    # corpus_position_tokens = set() # we want to see how sparse measure and position is
    for piece in corpus_reader:
        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))
        # measures_count = 0
        # measures_set = set()
        # for t in text_list:
        #     if t[0] == 'M':
        #         measures_set.add(t)
        #         measures_count += 1
        # max_measured_count = max(max_measured_count, measures_count)

        # corpus_measure_time_sig_tokens.update(measures_set)
        # if paras['position_method'] == 'event':
        #     positions = {t[1:] for t in text_list if t[0] == 'P'}
        # else:
        #     positions = {t.split(':')[-1] for t in text_list if t[0] == 'N'}
        # corpus_position_tokens.update(positions)
    # corpus_measure_time_sig_tokens = list(corpus_measure_time_sig_tokens)
    # corpus_position_tokens = list(corpus_position_tokens)

    summary_string = (
        f'Average tokens per piece: {sum(token_count_per_piece) / len(token_count_per_piece)}\n'\
        f'Event vocab size: {len(event_tokens)}\n'\
        f'- Shapes: {len(bpe_shapes_list)}\n'\
        f'- Tracks: {paras["max_track_number"]}\n'\
        f'- Measure/TimeSig event: {len(event_measure_time_sig_tokens)}\n'\
        f'- Position event: {len(position_tokens)}\n'\
        f'- Tempo event: {len(event_tempo_tokens)}\n'\
        f'Largest possible duration: {paras["max_duration"] * paras["nth"] // 4}\n'\
        f'Velocity vocab: {velocity_tokens}\n'\
        f'Tempo vocab: {len(event_tempo_tokens)} ({event_tempo_tokens})'\
    )
    # print(f'Vocabulary build time: {time()-start_time}')

    # build a dictionary for every token list
    vocabs = Vocabs(
        paras=paras,
        bpe_shapes_list=bpe_shapes_list,
        padding_token=PADDING_TOKEN_STR,
        events=event_tokens,
        pitchs=pitch_tokens,
        durations=duration_tokens,
        velocities=velocity_tokens,
        track_numbers=track_number_tokens,
        instruments=instrument_tokens,
        positions=position_tokens,
        tempos=event_tempo_tokens,
        time_signatures=event_measure_time_sig_tokens
    )

    return vocabs, summary_string
