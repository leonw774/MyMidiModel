import sys

from util import piece_to_midi, CorpusIterator, get_corpus_paras

def print_help():
    print('python3 text_to_midi.py corpus_dir_path out_path begin end')
    print('The output file name will be [out_path]_[i].mid, where i is the index number of piece in corpus.')
    print('If [being] and [end] are specified, will only out the pieces in range [begin, end)')
    print('If [end] is unset, it will be begin + 1.')
    print('If [end] is -1, it will be set to the length of corpus.')

def text_to_midi_read_args():
    if len(sys.argv) == 3:
        corpus_dir_path = sys.argv[1]
        out_path = sys.argv[2]
        begin = 0
        end = -1
    elif len(sys.argv) == 4:
        corpus_dir_path = sys.argv[1]
        out_path = sys.argv[2]
        begin = int(sys.argv[3])
        end = begin + 1
    elif len(sys.argv) == 5:
        corpus_dir_path = sys.argv[1]
        out_path = sys.argv[2]
        begin = int(sys.argv[3])
        end = int(sys.argv[4])
    else:
        print_help()
        exit()
    return corpus_dir_path, out_path, begin, end


def text_to_midi(corpus_dir_path, out_path, begin, end):
    with CorpusIterator(corpus_dir_path) as corpus_iterator:
        corpus_paras = get_corpus_paras(corpus_dir_path)
        print(corpus_paras)
        if len(corpus_paras) == 0:
            print('Error: no piece in input file')
            exit(1)
        if end == -1:
            end = len(corpus_iterator)
        # decode
        for i, piece in enumerate(corpus_iterator):
            if begin <= i < end:
                if len(piece) == 0:
                    continue
                # with open(f'{out_path}_{i}', 'w+', encoding='utf8') as tmp_file:
                #     tmp_file.write(piece)
                #     tmp_file.write('\n')
                midi = piece_to_midi(piece, corpus_paras['nth'], ignore_panding_note_error=False)
                midi.dump(f'{out_path}_{i}.mid')
                print(f'dumped {out_path}_{i}.mid')
            elif i >= end:
                break

if __name__ == '__main__':
    corpus_dir_path, out_path, begin, end = text_to_midi_read_args()
    print('start text_to_midi.py:', corpus_dir_path, out_path, begin, end)
    text_to_midi(corpus_dir_path, out_path, begin, end)