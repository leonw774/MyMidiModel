import io
import random
import sys

from miditoolkit import MidiFile

from data import file_to_corpus_paras_and_piece_iterator, piece_to_midi

def note_sort_key(n):
    return (n.start, n.pitch, n.end, n.velocity)

original_corpus_path = sys.argv[1]
tokenized_corpus_path = sys.argv[2]
sample_size = int(sys.argv[3])
assert sample_size > 0

orig_corpus_file = open(original_corpus_path, 'r', encoding='utf8')
tokn_corpus_file = open(tokenized_corpus_path, 'r', encoding='utf8')
orig_paras, orig_piece_iter = file_to_corpus_paras_and_piece_iterator(orig_corpus_file)
tokn_paras, tokn_piece_iter = file_to_corpus_paras_and_piece_iterator(tokn_corpus_file)
assert orig_paras == tokn_paras
nth = orig_paras['nth']

if sample_size > len(orig_piece_iter):
    sample_size = len(orig_piece_iter)
sample_indices = random.sample(
    list(range(len(orig_piece_iter))),
    sample_size
)
sample_indices = set(sample_indices)
orig_piece = None
tokn_piece = None
for i, (op, tp) in enumerate(zip(orig_piece_iter, tokn_piece_iter)):
    if i in sample_indices:
        orig_piece = op
        tokn_piece = tp
    else:
        continue

    try:
        orig_midi = piece_to_midi(orig_piece, nth)
    except Exception as e:
        print(f'exception in piece_to_midi of original piece #{i}')
        raise e

    try:
        tokn_midi = piece_to_midi(tokn_piece, nth)
    except Exception as e:
        print(f'exception in piece_to_midi of tokenized piece #{i}')
        raise e

    assert all(
        o_t.tempo == t_t.tempo and o_t.time == t_t.time
        for o_t, t_t in zip(orig_midi.tempo_changes, tokn_midi.tempo_changes) 
    ), f'tempos of piece #{i}'
    
    assert all(
        o_t.numerator == t_t.numerator and o_t.denominator == t_t.denominator and o_t.time == t_t.time
        for o_t, t_t in zip(orig_midi.time_signature_changes, tokn_midi.time_signature_changes) 
    ), f'time_signatures of piece #{i}'
    
    for j, (orig_inst, tokn_inst) in enumerate(zip(orig_midi.instruments, tokn_midi.instruments)):
        assert orig_inst.program == tokn_inst.program and orig_inst.is_drum == tokn_inst.is_drum, f'track #{j} of piece #{i}'
        sorted_orig_inst_notes = sorted(orig_inst.notes, key=note_sort_key)
        sorted_tokn_inst_notes = sorted(tokn_inst.notes, key=note_sort_key)
        different_notes = None
        for k, (o_n, t_n) in enumerate(zip(sorted_orig_inst_notes, sorted_tokn_inst_notes)):
            if not (o_n.velocity == t_n.velocity 
                    and o_n.pitch == t_n.pitch
                    and o_n.start == t_n.start
                    and o_n.end == t_n.end):
                different_notes = [sorted_orig_inst_notes[k-3:k+3], sorted_tokn_inst_notes[k-3:k+3]]
                break
        if different_notes is not None:
            # because sometime two or more overlapping notes of same pitch and velocity
            # if there is a continuing note in them, it will cause ambiguity in note merging
            orig_bytes_io = io.BytesIO()
            tokn_bytes_io = io.BytesIO()
            orig_midi.dump(file=orig_bytes_io)
            tokn_midi.dump(file=tokn_bytes_io)
            if orig_bytes_io.getvalue() != tokn_bytes_io.getvalue():
                raise AssertionError(f'notes #{k} of track #{j} of piece #{i}: {different_notes}')

orig_corpus_file.close()
tokn_corpus_file.close()

print("verification success")
exit(0)