import io
import random
import sys

from miditoolkit import MidiFile
from tqdm import tqdm

from data import quantize_to_nth, to_pathlist_file_path, CorpusIterator, get_corpus_paras,piece_to_midi

def verify_corpus_correctness(corpus_dir: str, sample_size: int) -> bool:
    """
        This function takes a corpus and check if its piece "equal" to its original midi file
        in a condition that "ticks per beat" of the original midi file is equal or less than the time step of the corpus.
    """

    assert sample_size > 0
    paras = get_corpus_paras(corpus_dir)
    midi_path_list = open(to_pathlist_file_path(corpus_dir), 'r', encoding='utf8').readlines()
    midi_path_list = [p[:-1] for p in midi_path_list] # there's '\n' at the end
    nth = paras['nth']

    with CorpusIterator(corpus_dir) as corpus_iterator:
        if sample_size > len(corpus_iterator):
            sample_size = len(corpus_iterator)
        sample_indices = random.sample(
            list(range(len(corpus_iterator))),
            sample_size
        )
        sample_indices = set(sample_indices)
        for i, piece in tqdm(enumerate(corpus_iterator), total=len(corpus_iterator)):
            if i not in sample_indices:
                continue

            original_midi = MidiFile(midi_path_list[i])
            quantize_to_nth(original_midi, nth)

            try:
                piece_midi = piece_to_midi(piece, nth)
            except Exception as e:
                print(f'exception in piece_to_midi of {corpus_dir} piece #{i}')
                raise e

            piece_bytes_io = io.BytesIO()
            piece_midi.dump(file=piece_bytes_io)

            # original_bytes = open(midi_path_list[i], 'rb').read()

            original_bytes_io = io.BytesIO()
            original_midi.dump(file=original_bytes_io)

            # if piece_bytes_io.getvalue() != original_bytes:
            if piece_bytes_io.getvalue() != original_bytes_io.getvalue():
                print(f'different midi output between original and corpus at piece#{i}')
                return False
    return True

if __name__ == '__main__':
    corpus_dir = sys.argv[1]
    sample_size = int(sys.argv[2])
    if verify_corpus_correctness(corpus_dir, sample_size):
        print("correctness verification success")
        exit(0)
    else:
        exit(1)
