import json
import logging
import os
import shutil
from argparse import ArgumentParser
from time import strftime, time

import numpy as np
from tqdm import tqdm

from util import (
    to_shape_vocab_file_path,
    to_vocabs_file_path,
    get_corpus_paras,
    CorpusIterator,
    build_vocabs,
    text_list_to_array,
    get_input_array_debug_string
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--bpe',
        type=int,
        default=0,
        help='The number of iteration the BPE algorithm did. Default is 0.\
            If the number is integer that greater than 0, it implicated that BPE was performed.'
    )
    parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
        help='Path to the log file. Default is empty, which means no logging would be performed.'
    )
    parser.add_argument(
        '--use-existed',
        action='store_true'
    )
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    parser.add_argument(
        'corpus_dir_path'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    loglevel = logging.INFO
    if args.log_file_path:
        logging.basicConfig(
            filename=args.log_file_path,
            filemode='a',
            level=loglevel,
            format='%(message)s',
        )
        console = logging.StreamHandler()
        console.setLevel(loglevel)
        logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(
            level=loglevel,
            format='%(message)s'
        )
    logging.info(strftime('==== text_to_array.py start at %Y%m%d-%H%M ===='))

    if not os.path.isdir(args.corpus_dir_path):
        logging.info('%s does not exist', args.corpus_dir_path)
        exit(1)

    bpe_shapes_list = []
    if args.bpe > 0:
        logging.info('Used BPE: %d', args.bpe)
        with open(to_shape_vocab_file_path(args.corpus_dir_path), 'r', encoding='utf8') as vocabs_file:
            bpe_shapes_list = vocabs_file.read().splitlines()

    if (    os.path.isfile(to_vocabs_file_path(args.corpus_dir_path))
        and os.path.isfile(os.path.join(args.corpus_dir_path, 'arrays.npz'))):
        logging.info('Corpus directory: %s already has vocabs file and array file.', args.corpus_dir_path)
        logging.info('Flag --use-existed is set')
        logging.info('==== text_to_array.py exited ====')
        return 1

    logging.info('Begin build vocabs for %s', args.corpus_dir_path)
    corpus_paras = get_corpus_paras(args.corpus_dir_path)
    with CorpusIterator(args.corpus_dir_path) as corpus_iterator:
        assert len(corpus_iterator) > 0, f'empty corpus: {args.corpus_dir_path}'

        vocabs, summary_string = build_vocabs(corpus_iterator, corpus_paras, bpe_shapes_list)
        with open(to_vocabs_file_path(args.corpus_dir_path), 'w+', encoding='utf8') as vocabs_file:
            json.dump(vocabs.to_dict(), vocabs_file)
        logging.info(summary_string)

        start_time = time()
        logging.info('Begin write npy')

        # handle existed files/dirs
        npy_dir_path = os.path.join(args.corpus_dir_path, 'arrays')
        npy_zip_path = os.path.join(args.corpus_dir_path, 'arrays.zip')
        npz_path = os.path.join(args.corpus_dir_path, 'arrays.npz')
        if os.path.exists(npy_dir_path):
            shutil.rmtree(npy_dir_path)
            print(f'Find existing {npy_dir_path}. Removed.')
        if os.path.exists(npy_zip_path):
            os.remove(npy_zip_path)
            print(f'Find existing {npy_zip_path}. Removed.')
        if os.path.exists(npz_path):
            os.remove(npz_path)
            print(f'Find existing {npz_path}. Removed.')

        os.makedirs(npy_dir_path)
        for i, p in tqdm(enumerate(corpus_iterator), total=len(corpus_iterator)):
            try:
                array = text_list_to_array(p.split(), vocabs)
            except Exception as e:
                print(f'piece #{i}')
                raise e
            np.save(os.path.join(npy_dir_path, str(i)), array)

        # zip all the npy files into one file with '.npz' extension
        shutil.make_archive(npy_dir_path, 'zip', root_dir=npy_dir_path)
        os.rename(npy_zip_path, npz_path)
        # delete the npys
        shutil.rmtree(npy_dir_path)

        # for debugging
        if args.debug:
            p0 = next(iter(corpus_iterator))
            original_text_list = p0.split()
            array_data = text_list_to_array(p0.split(), vocabs)

            debug_txt_path = os.path.join(args.corpus_dir_path, 'text_to_array_debug.txt')
            print(f'Write debug file: {debug_txt_path}')

            debug_str = get_input_array_debug_string(array_data, None, vocabs)
            debug_str_list = debug_str.splitlines()
            original_text_list = [f'{"original_text":<50} '] + original_text_list

            with open(debug_txt_path, 'w+', encoding='utf8') as f:
                merged_lines = [
                    f'{origi:<50} {debug}'
                    for origi, debug in zip(original_text_list, debug_str_list)
                ]
                f.write('\n'.join(merged_lines))

    logging.info('npys write time: %.3f', time()-start_time)
    logging.info('==== text_to_array.py exited ====')
    return 0

if __name__ == '__main__':
    main()
