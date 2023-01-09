# from util.corpus import CorpusReader # will cause circular import
from . import tokens
from .tokens import (
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

    # Events include:
    # - begin-of-score, end-of-score and seperator
    # - shape of multi-note/note
    # - track and instrument
    # - position
    # - measure and time_signature
    # - tempo change

    event_multi_note_shapes = (
        [tokens.NOTE_EVENTS_CHAR, tokens.NOTE_EVENTS_CHAR+'~']
        + [tokens.MULTI_NOTE_EVENTS_CHAR+shape for shape in bpe_shapes_list]
    )
    event_track_instrument = [tokens.TRACK_EVENTS_CHAR]
    event_position = (
        [tokens.POSITION_EVENTS_CHAR+int2b36str(i) for i in range(get_largest_possible_position(paras['nth']))]
        if paras['position_method'] == 'event' else
        []
    )
    event_measure_time_sig = [
        f'{tokens.MEASURE_EVENTS_CHAR}{int2b36str(n)}/{int2b36str(d)}' for n, d in SUPPORTED_TIME_SIGNATURES
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    event_tempo = [
        tokens.TEMPO_EVENTS_CHAR+int2b36str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]

    # padding token HAVE TO be at first
    event_vocab = (
        tokens.SPECIAL_TOKENS_STR + event_multi_note_shapes + event_track_instrument + event_position
        + event_measure_time_sig + event_tempo
    )

    pad_token = [tokens.PADDING_TOKEN_STR]
    pitch_vocab = pad_token + list(map(int2b36str, range(128)))
    duration_vocab = pad_token + list(map(int2b36str, range(1, paras['max_duration']+1))) # add 1 because range end is exclusive
    velocity_vocab = pad_token + list(map(int2b36str, range(paras['velocity_step']//2, 128, paras['velocity_step'])))
    track_number_vocab = pad_token + list(map(int2b36str, range(paras['max_track_number'])))
    instrument_vocab = pad_token + list(map(int2b36str, range(129)))

    # measure number are just increasing integer that generated dynamically in corpus.text_to_array, no vocab needed
    # position vocab need to use pure b36 int string because we have posattr setting
    position_vocab = pad_token + list(map(int2b36str, range(get_largest_possible_position(paras['nth']))))
    # these two vocabs are just same as their event counterparts just with PAD
    tempo_vocab = pad_token + event_tempo
    time_sig_vocab = pad_token + event_measure_time_sig

    token_count_per_piece = []
    for piece in corpus_reader:
        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))

    summary_string = (
        f'Average tokens per piece: {sum(token_count_per_piece) / len(token_count_per_piece)}\n'\
        f'Event vocab size: {len(event_vocab)}\n'\
        f'- Track number: {paras["max_track_number"]}\n'\
        f'- Measure-time signature: {len(event_measure_time_sig)}\n'\
        f'- Position: {len(position_vocab)}\n'\
        f'- Tempo: {len(event_tempo)} ({event_tempo})\n'\
        f'- Shape: {len(event_multi_note_shapes)}\n'\
        f'Duration vocab size: {len(duration_vocab)}\n'\
        f'Velocity vocab size: {len(velocity_vocab)} ({velocity_vocab})\n'\
    )
    # print(f'Vocabulary build time: {time()-start_time}')

    # build a dictionary for every token list
    vocabs = Vocabs(
        paras=paras,
        bpe_shapes_list=bpe_shapes_list,
        padding_token=tokens.PADDING_TOKEN_STR,
        events=event_vocab,
        pitchs=pitch_vocab,
        durations=duration_vocab,
        velocities=velocity_vocab,
        track_numbers=track_number_vocab,
        instruments=instrument_vocab,
        positions=position_vocab,
        tempos=tempo_vocab,
        time_signatures=time_sig_vocab
    )

    return vocabs, summary_string
