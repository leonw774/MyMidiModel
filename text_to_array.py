import io
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
    to_vocabs_json_file_path,
    get_corpus_paras,
    CorpusIterator,
    build_vocabs,
    text_list_to_array,
    array_to_text_list
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--max-sample-length',
        dest='max_sample_length',
        type=int,
        default=4096
    )
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
        '--debug',
        action='store_true'
    )
    parser.add_argument(
        'corpus_dir_path',
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
    logging.info(strftime('----text_to_array.py start at %Y%m%d-%H%M---'))

    if not os.path.isdir(args.corpus_dir_path):
        logging.info('%s does not exist', args.corpus_dir_path)
        exit(1)

    bpe_shapes_list = []
    if args.bpe > 0:
        logging.info('Used BPE: %d', args.bpe)
        with open(to_shape_vocab_file_path(args.corpus_dir_path), 'r', encoding='utf8') as vocabs_file:
            bpe_shapes_list = vocabs_file.read().splitlines()

    logging.info('Begin build vocabs for %s', args.corpus_dir_path)
    corpus_paras = get_corpus_paras(args.corpus_dir_path)
    with CorpusIterator(args.corpus_dir_path) as corpus_iterator:
        assert len(corpus_iterator) > 0, f'empty corpus: {args.corpus_dir_path}'

        vocab_dicts, summary_string = build_vocabs(corpus_iterator, corpus_paras, args.max_sample_length, bpe_shapes_list)
        with open(to_vocabs_json_file_path(args.corpus_dir_path), 'w+', encoding='utf8') as vocabs_file:
            json.dump(vocab_dicts, vocabs_file)
        logging.info(summary_string)

        start_time = time()
        logging.info('Begin write npy')

        # handle existed files/dirs
        npy_dir_path = os.path.join(args.corpus_dir_path, 'arrays')
        npy_zip_path = os.path.join(args.corpus_dir_path, 'arrays.zip')
        npz_path = os.path.join(args.corpus_dir_path, 'arrays.npz')
        if os.path.exists(npy_dir_path):
            shutil.rmtree(npy_dir_path)
            print(f'Find existing {npy_dir_path}, removed.')
        if os.path.exists(npy_zip_path):
            os.remove(npy_zip_path)
            print(f'Find existing {npy_zip_path}, removed.')
        if os.path.exists(npz_path):
            os.remove(npz_path)
            print(f'Find existing {npz_path}, removed.')

        os.makedirs(npy_dir_path)
        for i, p in tqdm(enumerate(corpus_iterator), total=len(corpus_iterator)):
            array = text_list_to_array(p.split(), vocab_dicts)
            np.save(os.path.join(npy_dir_path, str(i)), array)

        # zip all the npy files into one file with '.npz' extension
        shutil.make_archive(npy_dir_path, 'zip', root_dir=npy_dir_path)
        os.rename(npy_zip_path, npz_path)
        # delete the npys
        shutil.rmtree(npy_dir_path)

        # for debugging
        if args.debug:
            p0 = next(iter(corpus_iterator))
            array_data = text_list_to_array(p0.split(), vocab_dicts)
            array_text_byteio = io.BytesIO()
            debug_txt_path = os.path.join(args.corpus_dir_path, 'text_list_to_array_debug.txt')
            print(f'Write debug file: {debug_txt_path}')
            np.savetxt(array_text_byteio, array_data, fmt='%d')
            array_text_list = array_text_byteio.getvalue().decode().split('\n')
            original_text_list = p0.split()
            reconst_text_list = array_to_text_list(array_data, vocab_dicts)
            text_data_reconst_text_list = [
                (t, *d.split(), rt)
                for t, d, rt in zip(original_text_list, array_text_list, reconst_text_list)
            ]
            with open(debug_txt_path, 'w+', encoding='utf8') as f:
                f.write(
                    f"{'token_text':<50}{'evt':>5}{'pit':>5}{'dur':>5}{'vel':>5}"
                    + f"{'trn':>5}{'ins':>5}{'tmp':>5}{'pos':>5}{'mea':>5}     {'reconstructed_text'}\n")
                formated_lines = [
                    f"{line[0]:<50}{line[1]:>5}{line[2]:>5}{line[3]:>5}{line[4]:>5}{line[5]:>5}"
                    + f"{line[6]:>5}{line[7]:>5}{line[8]:>5}{line[9]:>5}     {line[10]}"
                    for line in text_data_reconst_text_list
                ]
                f.write('\n'.join(formated_lines))

    logging.info('npys write time: %.3f', time()-start_time)
    logging.info('make_data.py exited')

if __name__ == '__main__':
    main()
