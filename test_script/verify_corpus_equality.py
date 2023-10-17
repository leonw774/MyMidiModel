from collections import Counter
import io
from itertools import repeat
from multiprocessing import Pool
from psutil import cpu_count
import sys
from traceback import format_exc

from tqdm import tqdm

from util.midi import piece_to_midi
from util.corpus_reader import CorpusReader
from util.corpus import get_corpus_paras

def compare_two_pieces(piece_index: int, a_piece: str, b_piece: str, nth: int) -> bool:
    try:
        a_midi = piece_to_midi(a_piece, nth)
    except:
        print(f'exception in piece_to_midi of {a_corpus_dir} piece #{piece_index}')
        print(format_exc())
        # print('Piece:', a_piece)
        return False

    try:
        b_midi = piece_to_midi(b_piece, nth)
    except:
        print(f'exception in piece_to_midi of {b_corpus_dir} piece #{piece_index}')
        print(format_exc())
        # print('Piece:', b_piece)
        return False

    if any(
            o_t.tempo != t_t.tempo or o_t.time != t_t.time
            for o_t, t_t in zip(a_midi.tempo_changes, b_midi.tempo_changes)
        ):
        print(f'tempo difference at piece#{piece_index}')
        return False

    if any(
            o_t.numerator != t_t.numerator or o_t.denominator != t_t.denominator or o_t.time != t_t.time
            for o_t, t_t in zip(a_midi.time_signature_changes, b_midi.time_signature_changes)
        ):
        print(f'time signature difference at piece#{piece_index}')
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
        print(f'track mapping difference at piece#{piece_index}')
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
                # ab_note_start_union = a_track_note_starts.union(b_track_note_starts)
                # diff_note_starts = ab_note_start_union.difference(a_track_note_starts.intersection(b_track_note_starts))
                # diff_note_ends = ab_note_start_union.difference(a_track_note_ends.intersection(b_track_note_ends))
                # print(
                #     f'note starts diffs: {len(diff_note_starts)} / ({len(a_track_note_starts)} + {len(b_track_note_starts)})',
                #     f'\nnote end diffs: {len(diff_note_ends)} / ({len(a_track_note_ends)} + {len(b_track_note_ends)})\n'
                # )
                # print(f'diff_note_starts\n{diff_note_starts}')
                # print(f'diff_note_ends\n{diff_note_ends}', )
                # print("---")
                print(f'notes difference at piece#{piece_index}, track#{a_track_index_mapping[track_feature]}')
                return False
    return True

def compare_wrapper(args_dict) -> bool:
    return compare_two_pieces(**args_dict)

def verify_corpus_equality(a_corpus_dir: str, b_corpus_dir: str, mp_worker_number: int) -> bool:
    """
        Takes two corpus and check if they are "equal", that is,
        all the pieces at the same index are representing the same midi information,
        regardless of any token, shape or order differences.
    """

    paras = get_corpus_paras(a_corpus_dir)
    if paras != get_corpus_paras(b_corpus_dir):
        return False

    equality = True
    with CorpusReader(a_corpus_dir) as a_corpus_reader, CorpusReader(b_corpus_dir) as b_corpus_reader:
        assert len(a_corpus_reader) == len(b_corpus_reader)
        corpus_length = len(a_corpus_reader)
        nth = paras['nth']
        args_zip = zip(
            range(corpus_length),      # piece_index
            a_corpus_reader,         # a_piece
            b_corpus_reader,         # b_piece
            repeat(nth, corpus_length) # nth
        )
        args_dict_list = [
            {
                'piece_index': i,
                'a_piece': a,
                'b_piece': b,
                'nth': nth
            }
            for i, a, b, nth in args_zip
        ]

        with Pool(mp_worker_number) as p:
            for result in tqdm(p.imap_unordered(compare_wrapper, args_dict_list), total=corpus_length):
                if not result:
                    p.terminate()
                    equality = False
                    break
    return equality


if __name__ == '__main__':
    if len(sys.argv) == 3:
        a_corpus_dir = sys.argv[1]
        b_corpus_dir = sys.argv[2]
        mp_worker_number = min(cpu_count(), 4)
    elif len(sys.argv) == 4:
        a_corpus_dir = sys.argv[1]
        b_corpus_dir = sys.argv[2]
        mp_worker_number = int(sys.argv[3])
    else:
        print('Bad arguments')
        exit(1)
    print("Begin equality verification")
    if verify_corpus_equality(a_corpus_dir, b_corpus_dir, mp_worker_number):
        print("Equality verification success")
        exit(0)
    else:
        exit(1)
