import bisect
from collections import Counter
from fractions import Fraction
import functools
import itertools
from math import log, isnan, sqrt
from math import e as math_e
from multiprocessing import Pool
import random
from statistics import NormalDist
from typing import Dict, List, Union

import numpy as np
from miditoolkit import MidiFile
from pandas import Series
from tqdm import tqdm

from .tokens import b36strtoi, MEASURE_EVENTS_CHAR, END_TOKEN_STR
from .midi import piece_to_midi, midi_to_piece, get_after_k_measures


EVAL_SCALAR_FEATURE_NAMES = [
    'pitch_class_entropy',
    # 'pitchs_mean',
    # 'pitchs_var',
    # 'durations_mean',
    # 'durations_var',
    # 'velocities_mean',
    # 'velocities_var',
    'instrumentation_self_similarity',
    'grooving_self_similarity'
]

EVAL_DISTRIBUTION_FEATURE_NAMES = [
    'pitch_histogram',
    'duration_distribution',
    'velocity_histogram'
]


def _entropy(x: Union[list, dict], base: float = math_e) -> float:
    if len(x) == 0:
        raise ValueError()
    if isinstance(x, list):
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
    elif isinstance(x, dict):
        return _entropy(list(x.values()))
    else:
        raise TypeError('x should be both list or dict')



def kl_divergence(
        pred: dict,
        true: dict,
        base: float = math_e,
        ignore_pred_zero: bool = False) -> float:
    """Calculate KL(true || pred)"""
    if isinstance(pred, dict) and isinstance(true, dict):
        assert len(pred) > 0 and len(true) > 0
        if ignore_pred_zero:
            _true = {k: true[k] for k in true if k in pred}
        else:
            _true = true
        if len(_true) == 0:
            # uh-oh
            return float('nan')
        sum_pred = sum(pred.values())
        sum_true = sum(_true.values())
        if sum_pred == 0 or sum_true == 0:
            raise ValueError()
        norm_pred = {k: pred[k] / sum_pred for k in pred}
        norm_true = {k: _true[k] / sum_true for k in _true}
        kld = sum([
            norm_true[x] * log(norm_true[x] / norm_pred[x], base)
            if x in norm_pred else
            float('inf')
            for x in norm_true
        ])
        return kld
    else:
        raise TypeError('pred and true should be both dict')


def estimated_normal_overlapping_area(
        distribution_1: dict,
        distribution_2: dict) -> float:
    if isinstance(distribution_1, dict) and isinstance(distribution_2, dict):
        assert len(distribution_1) > 0 and len(distribution_2) > 0
        total_counts_1 = sum(
            count
            for count in distribution_1.values()
        )
        mean_1 = sum(
            number * count
            for number, count in distribution_1.items()
        ) / total_counts_1
        square_mean_1 = sum(
            number * number * count
            for number, count in distribution_1.items()
        ) / total_counts_1
        std_1 = sqrt(square_mean_1 - mean_1 * mean_1)

        total_counts_2 = sum(
            count
            for count in distribution_2.values()
        )
        mean_2 = sum(
            number * count
            for number, count in distribution_2.items()
        ) / total_counts_2
        square_mean_2 = sum(
            number * number * count
            for number, count in distribution_2.items()
        ) / total_counts_2
        std_2 = sqrt(square_mean_2 - mean_2 * mean_2)
    else:
        raise TypeError('the distributions should be both dict')
    if std_1 == 0 or std_2 == 0:
        return float('nan')
    nd_1 = NormalDist(mean_1, std_1)
    nd_2 = NormalDist(mean_2, std_2)
    return nd_1.overlap(nd_2)


def histogram_intersection(
        distribution_1: dict,
        distribution_2: dict) -> float:
    if isinstance(distribution_1, dict) and isinstance(distribution_2, dict):
        assert len(distribution_1) > 0 and len(distribution_2) > 0
        sum_1 = sum(distribution_1.values())
        sum_2 = sum(distribution_2.values())
        if sum_1 == 0 or sum_2 == 0:
            raise ValueError()
        norm_1 = {k: distribution_1[k] / sum_1 for k in distribution_1}
        norm_2 = {k: distribution_2[k] / sum_2 for k in distribution_2}
        all_keys = set(norm_1.keys()).union(set(norm_2.keys()))
        intersection = sum([
            min(norm_1.get(x, 0), norm_2.get(x, 0))
            for x in all_keys
        ])
        return intersection
    else:
        raise TypeError('the distributions should be both dict')


def random_sample_from_piece(piece: str, sample_measure_number: int):
    """
    Randomly sample a certain number of measures from a piece.
    Return sampled measures excerpt of the piece.
    """
    text_list = piece.split(' ')
    measure_indices = [
        i
        for i, t in enumerate(text_list)
        if t[0] == MEASURE_EVENTS_CHAR
    ]
    if len(measure_indices) < sample_measure_number:
        raise AssertionError(
            'piece has fewer measure than sample_measure_number'
        )

    start_index = random.randint(
        0,
        len(measure_indices) - sample_measure_number
    )
    if start_index + sample_measure_number < len(measure_indices):
        end_index = start_index + sample_measure_number
        sampled_body = text_list[start_index:end_index]
    else:
        sampled_body = text_list[start_index:]

    head = text_list[:measure_indices[0]]
    return head + sampled_body


EVALUATION_MIDI_TO_PIECE_PARAS_DEFAULT = {
    'tpq': 48, # 48 is default ticks-per-beat
    'max_track_number': 255, # limit of the multi-note bpe program
    'max_duration': 48 * 4, # four beats
    'velocity_step': 1,
    'tempo_quantization': (1, 512, 1),
    'use_cont_note': True,
}


def midi_to_features(
        midi: MidiFile,
        midi_to_piece_paras: Union[dict, None] = None,
        primer_measure_length: int = 0,
        max_pairs_number: int = int(1e6),
        max_token_number: int = int(1e4)
    ) -> Union[Dict[str, float], Exception]:
    """
    Return the feature the piece of the midi processed with given
    `midi_to_piece_paras`.

    If the `midi_to_piece_paras` check and `midi_to_piece` failed,
    this function return None.
    """

    if midi_to_piece_paras is None:
        midi_to_piece_paras = EVALUATION_MIDI_TO_PIECE_PARAS_DEFAULT
    else:
        # assert isinstance(midi_to_piece_paras, dict)
        if not isinstance(midi_to_piece_paras, dict):
            return ValueError('midi_to_piece_paras is not dict')
        if len(midi_to_piece_paras) == 0:
            midi_to_piece_paras = EVALUATION_MIDI_TO_PIECE_PARAS_DEFAULT
        else:
            has_all_keys = all(
                k in midi_to_piece_paras
                for k in EVALUATION_MIDI_TO_PIECE_PARAS_DEFAULT
            )
            # assert has_all_keys
            if has_all_keys:
                return ValueError('midi_to_piece_paras is not complete')
    tpq = midi_to_piece_paras['tpq']
    try:
        temp_piece = midi_to_piece(
            midi=midi,
            **midi_to_piece_paras,
            deny_long_empty_measures=False
        )
    except AssertionError as e:
        return e
    return piece_to_features(
        piece=temp_piece,
        tpq=tpq,
        primer_measure_length=primer_measure_length,
        max_pairs_number=max_pairs_number,
        max_token_number=max_token_number
    )


def piece_to_features(
        piece: str,
        tpq: int,
        primer_measure_length: int = 0,
        max_pairs_number: int = int(1e6),
        max_token_number: int = int(1e4)) -> Union[dict, Exception]:
    """
    Return a dict that contains:
    - pitch class histogram and entropy
    - duration distribution
    - velocity histogram
    - instrumentation self-similarity
    - grooving self-similarity

    Return None if piece_to_midi failed.
    """

    if primer_measure_length > 0:
        cut_piece = get_after_k_measures(
            piece.split(' '),
            primer_measure_length
        )
        piece = ' '.join(cut_piece)

    if max_token_number > 0:
        text_list = piece.split(' ')[:max_token_number]
        if text_list[-1] != END_TOKEN_STR:
            text_list.append(END_TOKEN_STR)
        piece = ' '.join(text_list)

    try:
        midi = piece_to_midi(piece, tpq, ignore_pending_note_error=True)
    except AssertionError as e:
        return e
    pitchs = []
    durations = []
    velocities = []

    for track in midi.instruments:
        for note in track.notes:
            pitchs.append(note.pitch)
            # use quarter note as unit for duration
            durations.append(Fraction((note.end - note.start), tpq))
            velocities.append(note.velocity)

    pitch_histogram = {p: 0 for p in range(128)}
    pitch_class_histogram = {pc: 0 for pc in range(12)}
    for p in pitchs:
        pitch_histogram[p] += 1
        pitch_class_histogram[p%12] += 1
    try:
        pitch_class_entropy = _entropy(pitch_class_histogram, base=2)
    except ValueError:
        pitch_class_entropy = float('nan')
    # pitchs_mean = np.mean(pitchs) if len(pitchs) > 0 else float('nan')
    # pitchs_var = np.var(pitchs) if len(pitchs) > 0 else float('nan')

    duration_distribution = dict(
        Counter([f'{d.numerator}/{d.denominator}' for d in durations])
    )
    # durations_float = [float(d) for d in durations]
    # durations_mean = (
    #     np.mean(durations_float)
    #     if len(durations) > 0 else
    #     float('nan')
    # )
    # durations_var = (
    #     np.var(durations_float)
    #     if len(durations) > 0 else
    #     float('nan')
    # )

    velocity_histogram = {v: 0 for v in range(128)}
    for v in velocities:
        velocity_histogram[v] += 1
    # velocities_mean = (
    #     np.mean(velocities)
    #     if len(velocities) > 0 else
    #     float('nan')
    # )
    # velocities_var = (
    #     np.var(velocities)
    #     if len(velocities) > 0 else
    #     float('nan')
    # )

    if isnan(pitch_class_entropy):
        return {
            'pitch_histogram': pitch_histogram,
            # 'pitchs_mean': pitchs_mean,
            # 'pitchs_var': pitchs_var,
            'duration_distribution': duration_distribution,
            # 'durations_mean': durations_mean,
            # 'durations_var': durations_var,
            'velocity_histogram': velocity_histogram,
            # 'velocities_mean': velocities_mean,
            # 'velocities_var': velocities_var,
            'pitch_class_entropy': pitch_class_entropy,
            'instrumentation_self_similarity': float('nan'),
            'grooving_self_similarity': float('nan')
        }

    measure_onsets = [0]
    cur_measure_length = 0
    max_measure_length = 0
    for text in piece.split(' '):
        if text[0] == MEASURE_EVENTS_CHAR:
            measure_onsets.append(measure_onsets[-1] + cur_measure_length)
            numer, denom = (b36strtoi(x) for x in text[1:].split('/'))
            cur_measure_length = 4 * tpq * numer // denom
            max_measure_length = max(max_measure_length, cur_measure_length)

    # because multi-note representation
    # some note events in a multi-note in a measure
    # but does not really start at that measure
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
    instrument_index_mapping = {
        program: idx
        for idx, program in enumerate(present_instrument)
    }
    bar_instrumentation = np.zeros(
        shape=(len(measure_onsets), instrument_number),
        dtype=np.bool8
    )
    grooving_per_bar = np.zeros(
        shape=(len(measure_onsets), max_position),
        dtype=np.bool8
    )
    for track in midi.instruments:
        for note in track.notes:
            # find measure index so that the measure has the largest onset
            # while smaller than note.start
            measure_index = bisect.bisect_right(measure_onsets, note.start) - 1
            program_number = instrument_index_mapping[track.program]
            bar_instrumentation[measure_index, program_number] |= True
            position = note.start - measure_onsets[measure_index]
            assert position < max_position
            grooving_per_bar[measure_index, position] |= True

    pairs = list(itertools.combinations(range(len(measure_onsets)), 2))
    if len(pairs) > max_pairs_number:
        pairs = random.sample(pairs, max_pairs_number)

    instrumentation_similarities = [
        1 - (
            sum(np.logical_xor(bar_instrumentation[a], bar_instrumentation[b]))
            / instrument_number
        )
        for a, b in pairs
    ]

    grooving_similarities = [
        1 - (
            sum(np.logical_xor(grooving_per_bar[a], grooving_per_bar[b]))
            / max_position
        )
        for a, b in pairs
    ]

    instrumentation_self_similarity = np.mean(instrumentation_similarities)
    grooving_self_similarity = np.mean(grooving_similarities)

    features = {
        'pitch_histogram': pitch_histogram,
        'pitch_class_entropy': pitch_class_entropy,
        # 'pitchs_mean': pitchs_mean,
        # 'pitchs_var': pitchs_var,
        'duration_distribution': duration_distribution,
        # 'durations_mean': durations_mean,
        # 'durations_var': durations_var,
        'velocity_histogram': velocity_histogram,
        # 'velocities_mean': velocities_mean,
        # 'velocities_var': velocities_var,
        'instrumentation_self_similarity': instrumentation_self_similarity,
        'grooving_self_similarity': grooving_self_similarity
    }
    return features


def compare_with_ref(source_eval_features: dict, reference_eval_features: dict):
    """Add KLD, OA, and HI of distributions into source eval features"""
    # KL Divergence
    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        source_eval_features[fname+'_KLD'] = kl_divergence(
            source_eval_features[fname],
            reference_eval_features[fname],
            ignore_pred_zero=True
        )

    # Overlapping area of estimated gaussian distribution
    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        # They keys in this metric have to be number
        # pitch & velocity are intergers, duration is fraction
        # just make them all floats
        source_eval_features[fname+'_OA'] = estimated_normal_overlapping_area(
            {
                float(Fraction(k)): v
                for k, v in source_eval_features[fname].items()
            },
            {
                float(Fraction(k)): v
                for k, v in reference_eval_features[fname].items()
            }
        )

    # (Normalized) Histogram intersection
    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        source_eval_features[fname+'_HI'] = histogram_intersection(
            source_eval_features[fname],
            reference_eval_features[fname]
        )


def aggregate_features(features_list: List[Dict[str, float]]) -> dict:
    aggr_eval_features = dict()

    # aggregate scalar features
    aggr_scalar_eval_features = {
        fname: [
            fs[fname]
            for fs in features_list
        ]
        for fname in EVAL_SCALAR_FEATURE_NAMES
    }
    for fname in EVAL_SCALAR_FEATURE_NAMES:
        f64_pandas_series = Series(
            aggr_scalar_eval_features[fname],
            dtype='float64'
        )
        fname_description = dict(
            f64_pandas_series.dropna().describe()
        )
        fname_description: Dict[str, np.float64]
        aggr_eval_features[fname] = {
            k: float(v) for k, v in fname_description.items()
        }

    # aggregate distribution features
    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        aggr_eval_features[fname] = dict(sum(
            [Counter(features[fname]) for features in features_list],
            Counter() # starting value of empty counter
        ))

    # aggregate others
    aggr_eval_features['notes_number_per_piece'] = [
        sum(features['pitch_histogram'].values())
        for features in features_list
    ]

    return aggr_eval_features


def midi_list_to_features(
        midi_list: List[MidiFile],
        midi_to_piece_paras: Union[dict, None] = None,
        primer_measure_length: int = 0,
        max_pairs_number: int = int(1e6),
        worker_number: int = 1,
        use_tqdm: bool = False):
    """
    Given a list of midis, returns:
    - A list of indices of processable uncorrupted midi
    - Their aggregated features
    """

    midi_to_features_partial = functools.partial(
        midi_to_features,
        midi_to_piece_paras=midi_to_piece_paras,
        primer_measure_length=primer_measure_length,
        max_pairs_number=max_pairs_number
    )
    eval_features_list: List[ Union[Dict[str, float], Exception] ] = []
    if worker_number > 1:
        with Pool(worker_number) as p:
            eval_features_list = list(tqdm(
                p.imap(midi_to_features_partial, midi_list),
                total=len(midi_list),
                ncols=80,
                desc='Calculating features',
                disable=not use_tqdm
            ))
    else:
        eval_features_list = list(tqdm(
            map(midi_to_features_partial, midi_list),
            total=len(midi_list),
            ncols=80,
            desc='Calculating features',
            disable=not use_tqdm
        ))

    # for e in eval_features_list:
    #     if isinstance(e, Exception):
    #         print(repr(e))

    uncorrupt_midi_indices_list = [
        i
        for i, _ in enumerate(midi_list)
        if not isinstance(eval_features_list[i], Exception)
    ]
    eval_features_list = [
        f
        for f in eval_features_list
        if not isinstance(f, Exception)
    ]
    aggr_eval_features = aggregate_features(eval_features_list)
    return uncorrupt_midi_indices_list, aggr_eval_features


def piece_list_to_features(
        piece_list: List[str],
        tpq: int,
        primer_measure_length: int = 0,
        max_pairs_number: int = int(1e6),
        worker_number: int = 1,
        use_tqdm: bool = False):
    """Given a list of pieces, returns their aggregated features"""
    piece_to_features_partial = functools.partial(
        piece_to_features,
        tpq=tpq,
        primer_measure_length=primer_measure_length,
        max_pairs_number=max_pairs_number
    )
    eval_features_list: List[ Dict[str, float] ] = []
    if worker_number > 1:
        with Pool(worker_number) as p:
            eval_features_list = list(tqdm(
                p.imap(piece_to_features_partial, piece_list),
                total=len(piece_list),
                ncols=80,
                disable=not use_tqdm
            ))
    else:
        eval_features_list = list(tqdm(
            map(piece_to_features_partial, piece_list),
            total=len(piece_list),
            ncols=80,
            disable=not use_tqdm
        ))
    eval_features_list = [f for f in eval_features_list if f is not None]
    aggr_eval_features = aggregate_features(eval_features_list)
    return aggr_eval_features
