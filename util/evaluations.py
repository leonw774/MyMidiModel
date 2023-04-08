import bisect
from collections import Counter
from fractions import Fraction
import itertools
from math import log, isnan
from math import e as math_e
import random
from typing import Dict

import numpy as np
from miditoolkit import MidiFile

from .tokens import get_largest_possible_position, get_supported_time_signatures, b36str2int, MEASURE_EVENTS_CHAR, END_TOKEN_STR
from .midi import piece_to_midi, midi_to_piece, get_after_k_measures


def _entropy(x: list, base: float = math_e) -> float:
    if len(x) == 0:
        raise ValueError()
    sum_x = sum(x)
    if sum_x == 0:
        raise ValueError()
    norm_x = [i / sum_x for i in x]
    entropy = -sum([
        i * log(i, base)
        for i in norm_x
        if i != 0
    ])
    return entropy


def kl_divergence(pred, true, base: float = math_e, ignore_pred_zero: bool = False):
    """
        Calculate KL(true || pred)
    """
    if isinstance(pred, list) and isinstance(true, list):
        assert len(pred) == len(true) and len(true) > 0
        sum_pred = sum(pred)
        sum_true = sum(true)
        if sum_pred == 0 or sum_true == 0:
            raise ValueError()
        norm_pred = [x / sum_pred for x in pred]
        norm_true = [x / sum_true for x in true]
        kld = sum([
            p * log(p/q)
            for p, q in zip(norm_true, norm_pred)
            if p != 0 and q != 0
        ])
        return kld
    elif isinstance(pred, dict) and isinstance(true, dict):
        assert len(true) > 0
        if ignore_pred_zero:
            true = {k: v for k, v in true.items() if k in pred}
        sum_pred = sum(pred.values())
        sum_true = sum(true.values())
        if sum_pred == 0 or sum_true == 0:
            raise ValueError()
        norm_pred = {k: pred[k]/sum_pred for k in pred}
        norm_true = {k: true[k]/sum_true for k in true}
        kld = sum([
            norm_true[x] * log(norm_true[x]/norm_pred[x], base)
            if x in norm_pred else
            float('inf')
            for x in norm_true
        ])
        return kld
    else:
        raise TypeError('pred and true should be both list or dict')


def overlapping_area(pred, true):
    raise NotImplementedError()


def random_sample_from_piece(piece: str, sample_measure_number: int):
    '''
        randomly sample a certain number of measures from a piece (in text list)
        Return:
        - sampled measures excerpt of the piece (in text list)
    '''
    text_list = piece.split(' ')
    measure_indices = [
        i
        for i, t in enumerate(text_list)
        if t[0] == MEASURE_EVENTS_CHAR
    ]
    if len(measure_indices) < sample_measure_number:
        raise AssertionError('piece has fewer measure than sample_measure_number')

    start_measure_index = random.randint(0, len(measure_indices) - sample_measure_number)
    if start_measure_index + sample_measure_number < len(measure_indices):
        sampled_body = text_list[start_measure_index : start_measure_index+sample_measure_number]
    else:
        sampled_body = text_list[start_measure_index :]

    head = text_list[:measure_indices[0]]
    return head + sampled_body


def midi_to_features(
        midi: MidiFile,
        primer_measure_length: int = 0,
        max_pairs_number: int = int(1e6),
        max_token_number: int = int(1e4)) -> Dict[str, float]:
    nth = midi.ticks_per_beat * 4
    temp_piece = midi_to_piece(
        midi,
        nth=nth,
        max_track_number=len(midi.instruments),
        max_duration=nth,
        velocity_step=1,
        use_cont_note=True,
        tempo_quantization=(1,4096,1),
        use_merge_drums=False
    )
    return piece_to_features(temp_piece, nth, primer_measure_length, max_pairs_number, max_token_number)


def piece_to_features(
        piece: str,
        nth: int,
        primer_measure_length: int = 0,
        max_pairs_number: int = int(1e6),
        max_token_number: int = int(1e4)) -> Dict[str, float]:
    '''
        Return:
        - pitch class histogram entropy
        - duration means and variant
        - velocity means and variant
        - instrumentation self-similarity
        - grooving self-similarity
    '''

    if primer_measure_length > 0:
        piece = ' '.join(get_after_k_measures(piece.split(' '), primer_measure_length))

    if max_token_number > 0:
        text_list = piece.split(' ')[:max_token_number]
        if text_list[-1] != END_TOKEN_STR:
            text_list.append(END_TOKEN_STR)
        piece = ' '.join(text_list)

    midi = piece_to_midi(piece, nth, ignore_pending_note_error=True)
    pitchs = []
    durations = []
    velocities = []

    for track in midi.instruments:
        for note in track.notes:
            pitchs.append(note.pitch)
            # use quarter note as unit for duration
            durations.append(Fraction((note.end - note.start) * 4, nth))
            velocities.append(note.velocity)

    pitch_histogram = [0 for _ in range(128)]
    pitch_class_histogram = [0 for _ in range(12)]
    for p in pitchs:
        pitch_histogram[p] += 1
        pitch_class_histogram[p%12] += 1
    try:
        pitch_class_entropy = _entropy(pitch_class_histogram, base=2)
    except ValueError:
        pitch_class_entropy = float('nan')
    pitchs_mean = np.mean(pitchs) if len(pitchs) > 0 else float('nan')
    pitchs_var = np.var(pitchs) if len(pitchs) > 0 else float('nan')

    duration_distribution = Counter([f'{d.numerator}/{d.denominator}' for d in durations])
    durations_as_float = [float(d) for d in durations]
    durations_mean = np.mean(durations_as_float) if len(durations) > 0 else float('nan')
    durations_var = np.var(durations_as_float) if len(durations) > 0 else float('nan')

    velocity_histogram = {v:0 for v in range(128)}
    for v in velocities:
        velocity_histogram[v] += 1
    velocities_mean = np.mean(velocities) if len(velocities) > 0 else float('nan')
    velocities_var = np.var(velocities) if len(velocities) > 0 else float('nan')

    if isnan(pitch_class_entropy):
        return {
            'pitch_histogram': pitch_histogram,
            'pitch_class_entropy': pitch_class_entropy,
            'pitchs_mean': pitchs_mean,
            'pitchs_var': pitchs_var,
            'duration_distribution': duration_distribution,
            'durations_mean': durations_mean,
            'durations_var': durations_var,
            'velocity_histogram': velocity_histogram,
            'velocities_mean': velocities_mean,
            'velocities_var': velocities_var,
            'instrumentation_self_similarity': float('nan'),
            'grooving_self_similarity': float('nan')
        }

    measure_onsets = [0]
    cur_measure_length = 0
    max_measure_length = 0
    for text in piece.split(' '):
        if text[0] == MEASURE_EVENTS_CHAR:
            measure_onsets.append(measure_onsets[-1] + cur_measure_length)
            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_length = round(nth * numer / denom)
            max_measure_length = max(max_measure_length, cur_measure_length)

    # because multi-note representation
    # some note events in a multi-note in a measure but does not really start at that measure
    end_note_onset = max(
        note.start
        for track in midi.instruments
        for note in track.notes
    )
    while measure_onsets[-1] < end_note_onset:
        measure_onsets.append(measure_onsets[-1] + cur_measure_length)

    max_position = max_measure_length
    present_instrument = {track.program for track in midi.instruments}
    instrument_number = len(present_instrument)
    instrument_index_mapping = {program: idx for idx, program in enumerate(present_instrument)}
    instrumentation_per_bar = np.zeros(shape=(len(measure_onsets), instrument_number), dtype=np.bool8)
    grooving_per_bar = np.zeros(shape=(len(measure_onsets), max_position), dtype=np.bool8)
    for track in midi.instruments:
        for note in track.notes:
            # find measure index so that the measure has the largest onset while smaller than note.start
            measure_index = bisect.bisect_right(measure_onsets, note.start) - 1
            instrumentation_per_bar[measure_index, instrument_index_mapping[track.program]] |= True
            position = note.start - measure_onsets[measure_index]
            assert position < max_position, (measure_index, measure_onsets[measure_index-1:measure_index+2], note.start)
            grooving_per_bar[measure_index, position] |= True

    pairs = list(itertools.combinations(range(len(measure_onsets)), 2))
    if len(pairs) > max_pairs_number:
        pairs = random.sample(pairs, max_pairs_number)

    instrumentation_similarities = [
        1 - (
            sum(np.logical_xor(instrumentation_per_bar[a], instrumentation_per_bar[b])) / instrument_number
        )
        for a, b in pairs
    ]

    grooving_similarities = [
        1 - (
            sum(np.logical_xor(grooving_per_bar[a], grooving_per_bar[b])) / max_position
        )
        for a, b in pairs
    ]

    instrumentation_self_similarity = np.mean(instrumentation_similarities)
    grooving_self_similarity = np.mean(grooving_similarities)

    features = {
        'pitch_histogram': pitch_histogram,
        'pitch_class_entropy': pitch_class_entropy,
        'pitchs_mean': pitchs_mean,
        'pitchs_var': pitchs_var,
        'duration_distribution': duration_distribution,
        'durations_mean': durations_mean,
        'durations_var': durations_var,
        'velocity_histogram': velocity_histogram,
        'velocities_mean': velocities_mean,
        'velocities_var': velocities_var,
        'instrumentation_self_similarity': instrumentation_self_similarity,
        'grooving_self_similarity': grooving_self_similarity
    }
    return features

EVAL_SCALAR_FEATURE_NAMES = [
    'pitch_class_entropy',
    'pitchs_mean',
    'pitchs_var',
    'durations_mean',
    'durations_var',
    'velocities_mean',
    'velocities_var',
    'instrumentation_self_similarity',
    'grooving_self_similarity'
]

EVAL_DISTRIBUTION_FEATURE_NAMES = [
    'pitch_histogram',
    'duration_distribution',
    'velocity_histogram'
]
