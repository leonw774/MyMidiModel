from collections import Counter
import io
import sys
from itertools import repeat
from multiprocessing import Pool
from traceback import format_exc

from psutil import cpu_count
from tqdm import tqdm

from util.midi import piece_to_midi
from util.corpus import CorpusReader, get_corpus_paras

def compare_two_pieces(
        piece_index: int,
        a_piece: str,
        b_piece: str,
        tpq: int) -> bool:
    try:
        a_midi = piece_to_midi(a_piece, tpq)
    except Exception:
        print(f'exception in piece_to_midi of a_piece #{piece_index}')
        print(format_exc())
        # print('Piece:', a_piece)
        return False

    try:
        b_midi = piece_to_midi(b_piece, tpq)
    except Exception:
        print(f'exception in piece_to_midi of b_piece #{piece_index}')
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
            o_t.numerator != t_t.numerator
            or o_t.denominator != t_t.denominator
            or o_t.time != t_t.time
            for o_t, t_t in zip(
                a_midi.time_signature_changes,
                b_midi.time_signature_changes
            )
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
        if ((a_track_note_starts != b_track_note_starts)
            or (a_track_note_ends != b_track_note_ends)):
            # sometimes there are multiple overlapping notes of
            # same pitch and velocity. if there is a continuing note in them,
            # they will cause ambiguity while merging
            a_bytes_io = io.BytesIO()
            b_bytes_io = io.BytesIO()
            a_midi.dump(file=a_bytes_io)
            b_midi.dump(file=b_bytes_io)
            if a_bytes_io.getvalue() != b_bytes_io.getvalue():
                print(
                    f'notes difference at piece#{piece_index}'
                    f'track#{a_track_index_mapping[track_feature]}'
                )
                return False
    return True

def compare_wrapper(args_dict) -> bool:
    return compare_two_pieces(**args_dict)

def verify_corpus_equality(
        a_corpus_dir: str,
        b_corpus_dir: str,
        worker_number: int) -> bool:
    """
    Takes two corpus and check if they are "equal", that is, all the
    pieces at the same index are representing the same midi
    nformation, regardless of any token, shape or order differences.
    """
    paras = get_corpus_paras(a_corpus_dir)
    if paras != get_corpus_paras(b_corpus_dir):
        return False

    equality = True
    with CorpusReader(a_corpus_dir) as a_corpus_reader, \
         CorpusReader(b_corpus_dir) as b_corpus_reader:
        assert len(a_corpus_reader) == len(b_corpus_reader)
        corpus_length = len(a_corpus_reader)
        tpq = paras['tpq']
        args_zip = zip(
            range(corpus_length),       # piece_index
            a_corpus_reader,            # a_piece
            b_corpus_reader,            # b_piece
            repeat(tpq, corpus_length)  # tpq
        )
        args_dict_list = [
            {
                'piece_index': i,
                'a_piece': a,
                'b_piece': b,
                'tpq': tpq
            }
            for i, a, b, tpq in args_zip
        ]

        with Pool(worker_number) as p:
            tqdm_compare = tqdm(
                p.imap_unordered(compare_wrapper, args_dict_list),
                total=corpus_length,
                ncols=80
            )
            for result in tqdm_compare:
                if not result:
                    p.terminate()
                    equality = False
                    break
    return equality


if __name__ == '__main__':
    if len(sys.argv) == 3:
        _a_corpus_dir = sys.argv[1]
        _b_corpus_dir = sys.argv[2]
        _worker_number = min(cpu_count(), 4)
    elif len(sys.argv) == 4:
        _a_corpus_dir = sys.argv[1]
        _b_corpus_dir = sys.argv[2]
        _worker_number = int(sys.argv[3])
    else:
        print('Bad arguments')
        exit(2)
    if verify_corpus_equality(_a_corpus_dir, _b_corpus_dir, _worker_number):
        exit(0)
    else:
        exit(1)
