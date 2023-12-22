from argparse import ArgumentParser, RawTextHelpFormatter

from util.midi import piece_to_midi
from util.corpus_reader import CorpusReader
from util.corpus import get_corpus_paras, piece_to_roll

def read_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        'corpus_dir_path',
        metavar='CORPUS_DIR_PATH',
        type=str
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT_PATH',
        type=str,
        help='The output file path will be "{OUTPUT_PATH}_{i}.mid", \
            where i is the index number of piece in corpus.'
    )
    parser.add_argument(
        '--indexing', '-i',
        type=str,
        required=True,
        help="""Required at least one indexing string.
An indexing string is in the form of "INDEX" or "BEGIN:END".
Former: extract piece at INDEX.
Latter: extract pieces from BEGIN (inclusive) to END (exclusive).
If any number A < 0, it will be replaced to CORPUS_LENGTH - A.
If BEGIN is empty, 0 will be used.
If END is empty, CORPUS_LENGTH will be used.
BEGIN and END can not be empty at the same time.
Multiple indexing strings are seperated by commas.
Example: --indexing ":2, 3:5, 7, -7, -5:-3, -2:"
"""
    )
    parser.add_argument(
        '--extract-midi', '--midi',
        action='store_true',
        help='Output midi file(s) stored in extracted pieces \
            with path name "{OUTPUT_PATH}_{i}.mid"'
    )
    parser.add_argument(
        '--extract-txt', '--txt',
        action='store_true',
        help='Output text file(s) containing text representation of \
            extracted pieces with path "{OUTPUT_PATH}_{i}.txt"'
    )
    parser.add_argument(
        '--extract-img', '--img',
        action='store_true',
        help='Output PNG file(s) of the pianoroll representation of \
            extracted pieces with path "{OUTPUT_PATH}_{i}.png"'
    )
    args = parser.parse_args()
    return args

def parse_index_string(index_str_list: list, corpus_length: int) -> set:
    indices_to_extract = set()
    for index_str in index_str_list:
        if ':' in index_str:
            s = index_str.split(':')
            assert len(s) == 2
            b, e = s
            assert not (b == '' and e == '')
            if b == '':
                b = 0
            if e == '':
                e = corpus_length
            b = int(b)
            e = int(e)
            if b < 0:
                b = corpus_length + b
            if e < 0:
                e = corpus_length + e
            # print(b, e)
            indices_to_extract.update(list(range(b, e))) # implies b <= e
        else:
            i = int(index_str)
            if i < 0:
                i = corpus_length + i
            # print(i)
            indices_to_extract.add(i)
    return set(indices_to_extract)


def main():
    print('==== start extract.py ====')
    args = read_args()

    if not (args.extract_img or args.extract_txt or args.extract_midi):
        print('Please choose at least one format to output')
        return 0

    print('\n'.join([
        f'{k}: {v}'
        for k, v in vars(args).items()
    ]))

    with CorpusReader(args.corpus_dir_path) as corpus_reader:
        corpus_paras = get_corpus_paras(args.corpus_dir_path)
        print('Corpus parameters:')
        print(corpus_paras)
        if len(corpus_reader) == 0:
            print('Error: no piece in input file')
            return 1

        indices_to_extract = parse_index_string(
            index_str_list=args.indexing.split(','),
            corpus_length=len(corpus_reader)
        )
        print("extracting indices:", list(indices_to_extract))

        # extract
        for i in indices_to_extract:
            piece = corpus_reader[i]
            if len(piece) == 0:
                continue

            if args.extract_midi:
                midi = piece_to_midi(
                    piece,
                    corpus_paras['tpq'],
                    ignore_pending_note_error=False
                )
                midi.dump(f'{args.output_path}_{i}.mid')

            if args.extract_txt:
                output_file_path = f'{args.output_path}_{i}.txt'
                with open(output_file_path, 'w+', encoding='utf8') as f:
                    f.write(piece+'\n')

            if args.extract_img:
                figure = piece_to_roll(piece, corpus_paras['tpq'])
                figure.savefig(f'{args.output_path}_{i}.png')

            print(f'extracted {args.output_path}_{i}')

if __name__ == '__main__':
    EXIT_CODE = main()
    exit(EXIT_CODE)
