import re
from tqdm import tqdm

from . import tokens
from .tokens import (
    int2b36str,
    get_largest_possible_position,
    get_supported_time_signatures
)
from .corpus_reader import CorpusReader

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
                raise TypeError(f'AttrVocab\'s init_obj can only be list or dict. Got {type(init_obj)}')
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
    max_mps_number: int
    tempos: AttrVocab
    time_signatures: AttrVocab

    def __init__(self, paras: dict, bpe_shapes_list: list, padding_token: str,
            events, pitchs, durations, velocities, track_numbers, instruments,
            max_mps_number, tempos, time_signatures) -> None:
        self.paras = paras
        self.bpe_shapes_list = bpe_shapes_list
        self.padding_token = padding_token
        self.events = Vocabs.AttrVocab(events)
        self.pitchs = Vocabs.AttrVocab(pitchs)
        self.durations = Vocabs.AttrVocab(durations)
        self.velocities = Vocabs.AttrVocab(velocities)
        self.track_numbers = Vocabs.AttrVocab(track_numbers)
        self.instruments = Vocabs.AttrVocab(instruments)
        self.max_mps_number = max_mps_number
        self.tempos = Vocabs.AttrVocab(tempos)
        self.time_signatures = Vocabs.AttrVocab(time_signatures)

    @classmethod
    def from_dict(cls, d):
        return cls(
            d['paras'], d['bpe_shapes_list'], d['padding_token'],
            d['events'], d['pitchs'], d['durations'], d['velocities'], d['track_numbers'],
            d['instruments'], d['max_mps_number'], d['tempos'], d['time_signatures']
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
            'max_mps_number': self.max_mps_number,
            'tempos': vars(self.tempos),
            'time_signatures': vars(self.time_signatures),
        }


def build_vocabs(
        corpus_reader: CorpusReader,
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
    # - track
    # - position
    # - measure and time_signature
    # - tempo change

    supported_time_signatures = get_supported_time_signatures()
    largest_possible_position = get_largest_possible_position(paras['nth'], supported_time_signatures)

    event_multi_note_shapes = (
        [tokens.NOTE_EVENTS_CHAR, tokens.NOTE_EVENTS_CHAR+'~']
        + [tokens.MULTI_NOTE_EVENTS_CHAR+shape for shape in bpe_shapes_list]
    )
    event_track_instrument = (
        [tokens.TRACK_EVENTS_CHAR+int2b36str(i) for i in range(129)]
    )
    event_position = (
        [tokens.POSITION_EVENTS_CHAR+int2b36str(i) for i in range(largest_possible_position)]
    )
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    event_tempo = [
        tokens.TEMPO_EVENTS_CHAR+int2b36str(t) for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]
    event_measure_time_sig = [
        f'{tokens.MEASURE_EVENTS_CHAR}{int2b36str(n)}/{int2b36str(d)}' for n, d in supported_time_signatures
    ]

    measure_substr = ' '+tokens.MEASURE_EVENTS_CHAR
    position_substr = ' '+tokens.POSITION_EVENTS_CHAR
    tempo_substr = ' '+tokens.TEMPO_EVENTS_CHAR
    max_mps_number = 0
    corpus_measure_time_sigs = set()
    token_count_per_piece = []
    for piece in tqdm(corpus_reader):
        measure_number = piece.count(measure_substr)
        position_number = piece.count(position_substr)
        tempo_number = piece.count(tempo_substr)
        piece_max_mps_number = 2 * measure_number + 2 * position_number + tempo_number + 4 # +4 because head (3) and eos (1)
        max_mps_number = max(max_mps_number, piece_max_mps_number)

        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))
        corpus_measure_time_sigs.update([text for text in text_list if text[0] == tokens.MEASURE_EVENTS_CHAR])
    # should we remove time signatures that dont appears in target corpus?
    # event_measure_time_sig = [t for t in event_measure_time_sig if t in corpus_measure_time_sigs]

    # padding token HAVE TO be at first
    event_vocab = (
        tokens.SPECIAL_TOKENS_STR + event_multi_note_shapes + event_track_instrument + event_position
        + event_measure_time_sig + event_tempo
    )

    pad_token = [tokens.PADDING_TOKEN_STR]
    pitch_vocab = pad_token + list(map(int2b36str, range(128)))
    duration_vocab = pad_token + list(map(int2b36str, range(1, paras['max_duration']+1))) # add 1 because range end is exclusive
    velocity_vocab = pad_token + list(map(int2b36str, range(paras['velocity_step'], 128, paras['velocity_step'])))
    track_number_vocab = pad_token + list(map(int2b36str, range(paras['max_track_number'])))
    instrument_vocab = pad_token + list(map(int2b36str, range(129)))
    # mps number and measure number are just increasing integer that generated dynamically in corpus.text_to_array
    tempo_vocab = pad_token + event_tempo
    time_sig_vocab = pad_token + event_measure_time_sig

    summary_string = (
        f'Average tokens per piece: {sum(token_count_per_piece) / len(token_count_per_piece)}\n'\
        f'Event vocab size: {len(event_vocab)}\n'\
        f'- Measure-time signature: {len(event_measure_time_sig)} (Existed in corpus: {len(corpus_measure_time_sigs)})\n'\
        f'- Position: {len(event_position)}\n'\
        f'- Tempo: {len(event_tempo)} ({event_tempo})\n'\
        f'- Shape: {len(event_multi_note_shapes)}\n'\
        f'Duration vocab size: {len(duration_vocab)}\n'\
        f'Velocity vocab size: {len(velocity_vocab)} ({velocity_vocab})\n'\
        f'Track number vocab size: {paras["max_track_number"]}\n'\
        f'Max MPS number: {max_mps_number}'
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
        max_mps_number=max_mps_number,
        tempos=tempo_vocab,
        time_signatures=time_sig_vocab
    )

    return vocabs, summary_string
