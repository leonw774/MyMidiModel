import io
import random
import sys
from collections import Counter

from tqdm import tqdm

from util import CorpusIterator, get_corpus_paras, piece_to_midi

def verify_corpus_equality(a_corpus_dir: str, b_corpus_dir: str, sample_size) -> bool:
    """
        This function takes two corpus and check if they are "equal",
        that is, all the pieces at the same index are representing the same midi information,
        regardless of any token, shape or order differences.
    """

    paras = get_corpus_paras(a_corpus_dir)
    if paras != get_corpus_paras(b_corpus_dir):
        return False

    with CorpusIterator(a_corpus_dir) as a_corpus_iterator, CorpusIterator(b_corpus_dir) as b_corpus_iterator:
        nth = paras['nth']

        if sample_size is None:
            sample_size = len(a_corpus_iterator)
            sample_indices = set(range(sample_size))
        else:
            if sample_size >= len(a_corpus_iterator):
                sample_size = len(a_corpus_iterator)
                assert sample_size > 0
                sample_indices = set(range(sample_size))
            else:
                sample_indices = random.sample(
                    list(range(len(a_corpus_iterator))),
                    sample_size
                )
                sample_indices = set(sample_indices)

        for i, (a, b) in tqdm(enumerate(zip(a_corpus_iterator, b_corpus_iterator)), total=len(a_corpus_iterator)):
            if i in sample_indices:
                a_piece = a
                b_piece = b
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

            if any(
                    o_t.tempo != t_t.tempo or o_t.time != t_t.time
                    for o_t, t_t in zip(a_midi.tempo_changes, b_midi.tempo_changes)
                ):
                return False


            if any(
                    o_t.numerator != t_t.numerator or o_t.denominator != t_t.denominator or o_t.time != t_t.time
                    for o_t, t_t in zip(a_midi.time_signature_changes, b_midi.time_signature_changes)
                ):
                return False

            a_track_index_mapping = {
                (track.program, track.is_drum, len(track.notes)): i
                for i, track in enumerate(a_midi.instruments)
            }
            b_track_index_mapping = {
                (track.program, track.is_drum, len(track.notes)): i
                for i, track in enumerate(b_midi.instruments)
            }
            a_tracks_counter = dict(Counter(a_track_index_mapping.keys()))
            b_tracks_counter = dict(Counter(b_track_index_mapping.keys()))
            if a_tracks_counter != b_tracks_counter:
                return False

            for track_feature in a_track_index_mapping.keys():
                a_track = a_midi.instruments[a_track_index_mapping[track_feature]]
                b_track = b_midi.instruments[b_track_index_mapping[track_feature]]
                a_track_note_starts = frozenset({
                    (n.start, n.pitch, n.velocity)
                    for n in a_track.notes
                })
                a_track_note_ends = frozenset({
                    (n.end, n.pitch, n.velocity)
                    for n in a_track.notes
                })
                b_track_note_starts = frozenset({
                    (n.start, n.pitch, n.velocity)
                    for n in b_track.notes
                })
                b_track_note_ends = frozenset({
                    (n.end, n.pitch, n.velocity)
                    for n in b_track.notes
                })
                if (a_track_note_starts != b_track_note_starts) or (a_track_note_ends != b_track_note_ends):
                    # because sometimes there are two or more overlapping notes of same pitch and velocity
                    # if there is a continuing note in them, they will cause ambiguity in note merging
                    a_bytes_io = io.BytesIO()
                    b_bytes_io = io.BytesIO()
                    a_midi.dump(file=a_bytes_io)
                    b_midi.dump(file=b_bytes_io)
                    if a_bytes_io.getvalue() != b_bytes_io.getvalue():
                        # diff_note_starts = (a_track_note_starts.union(b_track_note_starts)).difference(a_track_note_starts.intersection(b_track_note_starts))
                        # diff_note_ends = (a_track_note_ends.union(b_track_note_ends)).difference(a_track_note_ends.intersection(b_track_note_ends))
                        # print(
                        #     f'notes starts difference non-empty: {len(diff_note_starts)} / ({len(a_track_note_starts)} + {len(b_track_note_starts)})\n'\
                        #     f'notes end difference non-empty: {len(diff_note_ends)} / ({len(a_track_note_ends)} + {len(b_track_note_ends)})\n'\
                        #     f'piece#{i}, track#{a_track_index_mapping[track_feature]}'\
                        # )
                        # print('diff_note_starts')
                        # print(diff_note_starts)
                        # print('diff_note_ends')
                        # print(diff_note_ends)
                        # print("---")
                        return False
    return True

if __name__ == '__main__':
    if len(sys.argv) == 3:
        a_corpus_dir = sys.argv[1]
        b_corpus_dir = sys.argv[2]
        sample_size = None
    elif len(sys.argv) == 4:
        a_corpus_dir = sys.argv[1]
        b_corpus_dir = sys.argv[2]
        sample_size = int(sys.argv[3])
    print("begin equality verification")
    if verify_corpus_equality(a_corpus_dir, b_corpus_dir, sample_size):
        print("equality verification success")
        exit(0)
    else:
        exit(1)
