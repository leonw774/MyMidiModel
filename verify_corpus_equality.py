import io
import random
import sys
from collections import Counter

from data import CorpusIterator, get_corpus_paras, piece_to_midi


def verify_corpus_equality(a_corpus_dir: str, b_corpus_dir: str, sample_size: int) -> bool:
    """
        This function takes two corpus and check if they are "equal",
        namely all the pieces in the same order are representing the same midi information,
        regardless of any token, shape or order differences.
    """

    assert sample_size > 0
    paras = get_corpus_paras(a_corpus_dir)
    if paras != get_corpus_paras(b_corpus_dir):
        return False

    with CorpusIterator(a_corpus_dir) as a_corpus_iterator, CorpusIterator(b_corpus_dir) as b_corpus_iterator:
        nth = paras['nth']

        if sample_size > len(a_corpus_iterator):
            sample_size = len(a_corpus_iterator)
        sample_indices = random.sample(
            list(range(len(a_corpus_iterator))),
            sample_size
        )
        sample_indices = set(sample_indices)
        for i, (op, tp) in enumerate(zip(a_corpus_iterator, b_corpus_iterator)):
            if i in sample_indices:
                a_piece = op
                b_piece = tp
            else:
                continue

            try:
                a_midi = piece_to_midi(a_piece, nth)
            except Exception as e:
                print(f'exception in piece_to_midi of {a_corpus_dir} piece #{i}')
                raise e

            try:
                b_midi = piece_to_midi(b_piece, nth)
            except Exception as e:
                print(f'exception in piece_to_midi of {b_corpus_dir} piece #{i}')
                raise e

            if any( o_t.qpm != t_t.qpm or o_t.time != t_t.time
                    for o_t, t_t in zip(a_midi.tempos, b_midi.tempos) ):
                return False


            if any( o_t.numerator != t_t.numerator or o_t.denominator != t_t.denominator or o_t.time != t_t.time
                    for o_t, t_t in zip(a_midi.time_signatures, b_midi.time_signatures) ):
                return False

            a_track_index_mapping = {
                (track.program, track.is_drum): i
                for i, track in enumerate(a_midi.tracks)
            }
            b_track_index_mapping = {
                (track.program, track.is_drum): i
                for i, track in enumerate(b_midi.tracks)
            }
            a_tracks_counter = dict(Counter(a_track_index_mapping.keys()))
            b_tracks_counter = dict(Counter(b_track_index_mapping.keys()))
            if a_tracks_counter != b_tracks_counter:
                return False

            for track_feature in a_track_index_mapping.keys():
                a_track = a_midi.tracks[a_track_index_mapping[track_feature]]
                b_track = b_midi.tracks[b_track_index_mapping[track_feature]]
                a_track_notes = frozenset(a_track.notes)
                b_track_notes = frozenset(b_track.notes)
                if a_track_notes != b_track_notes:
                    # because sometimes there are two or more overlapping notes of same pitch and velocity
                    # if there is a continuing note in them, they will cause ambiguity in note merging
                    a_bytes_io = io.BytesIO()
                    b_bytes_io = io.BytesIO()
                    a_midi.dump(file=a_bytes_io)
                    b_midi.dump(file=b_bytes_io)
                    if a_bytes_io.getvalue() != b_bytes_io.getvalue():
                        print(
                            f'notes difference non-empty: {a_track_notes.difference(b_track_notes)}\n'
                            f'{a_corpus_dir}: piece#{a_piece}, track#{a_track_index_mapping[track_feature]}\n'\
                            f'{b_corpus_dir}: piece#{b_piece}, track#{b_track_index_mapping[track_feature]}'\
                        )
                        return False
    return True

if __name__ == '__main__':
    a_corpus_dir = sys.argv[1]
    b_corpus_dir = sys.argv[2]
    sample_size = int(sys.argv[3])
    if verify_corpus_equality(a_corpus_dir, b_corpus_dir, sample_size):
        print("equality verification success")
        exit(0)
    else:
        exit(1)
