from argparse import ArgumentParser 

from util.midi import piece_to_midi
from util.corpus import CorpusIterator, get_corpus_paras

def text_to_midi_read_args():
    parser = ArgumentParser()
    parser.add_argument(
        'corpus_dir_path',
        type=str
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='The output file path will be "{OUTPUT_PATH}_{i}.mid", where i is the index number of piece in corpus.'
    )
    parser.add_argument(
        '--begin', '-b',
        type=int,
        default=0,
        help='The beginning index of outputed pieces. Default is 0.'
    )
    parser.add_argument(
        '--end', '-e',
        nargs='?',
        type=int,
        const=-1,
        help='If --end is set with specified value, the program output the pieces indexed in range [BEGIN, END)\n\
            If --end is set with no specified value, it will be set to the length of the corpus\n\
            If --end is not set, it will be set to [BEGIN]+1'
    )
    parser.add_argument(
        '--extract-txt',
        action='store_true',
        help='If set, will output additional files containing texts of each pieces with path name "{OUTPUT_PATH}_{i}.txt"'
    )
    args = parser.parse_args()
    if args.end is None:
        args.end = args.begin + 1
    return args.corpus_dir_path, args.output_path, args.begin, args.end, args.extract_txt


def text_to_midi(corpus_dir_path, out_path, begin, end, extract_txt):
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
                if extract_txt:
                    with open(f'{out_path}_{i}.txt', 'w+', encoding='utf8') as tmp_file:
                        tmp_file.write(piece)
                        tmp_file.write('\n')
                midi = piece_to_midi(piece, corpus_paras['nth'], ignore_panding_note_error=False)
                midi.dump(f'{out_path}_{i}.mid')
                print(f'dumped {out_path}_{i}.mid')
            elif i >= end:
                break

if __name__ == '__main__':
    corpus_dir_path, out_path, begin, end, extract_txt = text_to_midi_read_args()
    print('start text_to_midi.py:', corpus_dir_path, out_path, begin, end)
    text_to_midi(corpus_dir_path, out_path, begin, end, extract_txt)
