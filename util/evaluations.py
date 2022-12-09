import bisect
import itertools
from math import log, isnan
import random
from typing import Dict

import numpy as np

from .tokens import get_largest_possible_position, b36str2int
from .midi import piece_to_midi


def _entropy(x: list) -> float:
    if len(x) == 0:
        raise ValueError()
    sum_x = sum(x)
    if sum_x == 0:
        raise ValueError()
    norm_x = [i / sum_x for i in x]
    entropy = -sum([
        i * log(i)
        for i in norm_x
        if i != 0
    ])
    return entropy


def _kl_divergence(true, pred):
    if isinstance(true, list) and isinstance(pred, list):
        assert len(true) == len(pred) and len(true) > 0
        sum_true = sum(true)
        sum_pred = sum(pred)
        if sum_true == 0 or sum_pred == 0:
            raise ValueError()
        norm_true = [x / sum_true for x in true]
        norm_pred = [x / sum_pred for x in pred]
        entropy = -sum([
            p * log(q/p)
            for p, q in zip(norm_true, norm_pred)
            if p != 0 and q != 0
        ])
        return entropy
    elif isinstance(true, dict) and isinstance(pred, dict):
        assert len(true) > 0
        assert all((x in true) for x in pred), 'true(x) = 0 does not imply pred(x) = 0'
        sum_true = sum(true.values())
        sum_pred = sum(pred.values())
        if sum_true == 0 or sum_pred == 0:
            raise ValueError()
        norm_true = {k: true[k]/sum_true for k in true}
        norm_pred = {k: pred[k]/sum_pred for k in pred}
        entropy = -sum([
            norm_true[x] * log(norm_pred[x]/norm_true[x])
            for x in norm_true
            if norm_true[x] != 0 and x in norm_pred
        ])
        return entropy
    else:
        raise TypeError('pred and true should be both list or dict')


def _overlapping_area(true, pred):
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
        if t[0] == 'M'
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


def piece_to_features(piece: str, nth: int, max_pairs_number: int) -> Dict[str, float]:
    '''
        Return:
        - pitch class histogram entropy
        - duration means and variant
        - velocity means and variant
        - instrumentation self-similarity
        - grooving self-similarity
    '''
    midi = piece_to_midi(piece, nth, ignore_pending_note_error=True)
    pitch_histogram = [0] * 128
    durations = []
    velocities = []
    for track in midi.instruments:
        for note in track.notes:
            pitch_histogram[note.pitch] += 1
            durations.append(note.end - note.start)
            velocities.append(note.velocity)

    try:
        pitch_histogram_entropy = _entropy(pitch_histogram)
    except ValueError:
        pitch_histogram_entropy = float('nan')
    durations_mean = np.mean(durations) if len(durations) > 0 else float('nan')
    durations_var = np.var(durations) if len(durations) > 0 else float('nan')
    velocities_mean = np.mean(velocities) if len(velocities) > 0 else float('nan')
    velocities_var = np.var(velocities) if len(velocities) > 0 else float('nan')

    if isnan(pitch_histogram_entropy):
        return {
            'pitch_histogram_entropy': pitch_histogram_entropy,
            'durations_mean': durations_mean,
            'durations_var': durations_var,
            'velocities_mean': velocities_mean,
            'velocities_var': velocities_var,
            'instrumentation_self_similarity': float('nan'),
            'grooving_self_similarity': float('nan')
        }

    measure_onsets = [0]
    cur_measure_length = 0
    for text in piece.split(' '):
        if text[0] == 'M':
            measure_onsets.append(measure_onsets[-1] + cur_measure_length)
            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_length = round(nth * numer / denom)

    # because sometime the last note "in" the measure does not really IN the measure
    end_note_onset = max(
        note.start
        for track in midi.instruments
        for note in track.notes
    )
    while measure_onsets[-1] < end_note_onset:
        measure_onsets.append(measure_onsets[-1] + cur_measure_length)

    max_position = get_largest_possible_position(nth)
    instrumentation_per_bar = np.zeros(shape=(len(measure_onsets), 129), dtype=np.bool8)
    grooving_per_bar = np.zeros(shape=(len(measure_onsets), max_position), dtype=np.bool8)
    for track in midi.instruments:
        for note in track.notes:
            # find measure index so that the measure has the largest onset while smaller than note.start
            measure_index = bisect.bisect_right(measure_onsets, note.start) - 1
            instrumentation_per_bar[measure_index, track.program] |= True
            position = note.start - measure_onsets[measure_index]
            assert position < max_position, (measure_index, measure_onsets[measure_index-1:measure_index+2], note.start)
            grooving_per_bar[measure_index, position] |= True

    pairs = list(itertools.combinations(range(len(measure_onsets)), 2))
    if len(pairs) > max_pairs_number:
        pairs = random.sample(pairs, max_pairs_number)

    cross_instrumentation_similarities = [
        1 - (
            sum(np.logical_xor(instrumentation_per_bar[a], instrumentation_per_bar[b])) / 129
        )
        for a, b in pairs
    ]

    cross_grooving_similarities = [
        1 - (
            sum(np.logical_xor(grooving_per_bar[a], grooving_per_bar[b])) / max_position
        )
        for a, b in pairs
    ]

    instrumentation_self_similarity = np.mean(cross_instrumentation_similarities)
    grooving_self_similarity = np.mean(cross_grooving_similarities)

    features = {
        'pitch_histogram_entropy': pitch_histogram_entropy,
        'durations_mean': durations_mean,
        'durations_var': durations_var,
        'velocities_mean': velocities_mean,
        'velocities_var': velocities_var,
        'instrumentation_self_similarity': instrumentation_self_similarity,
        'grooving_self_similarity': grooving_self_similarity
    }
    return features

EVAL_FEATURE_NAMES = {
    'pitch_histogram_entropy',
    'durations_mean',
    'durations_var',
    'velocities_mean',
    'velocities_var',
    'instrumentation_self_similarity',
    'grooving_self_similarity'
}
