from collections import Counter
import json
import logging
import os
import shutil
from argparse import ArgumentParser
from time import strftime, time

import numpy as np
from tqdm import tqdm

from util.tokens import b36str2int, INSTRUMENT_NAMES
from util.vocabs import build_vocabs
from util.corpus import (
    to_shape_vocab_file_path,
    to_vocabs_file_path,
    get_corpus_paras,
    CorpusIterator,
    text_list_to_array,
    get_input_array_debug_string,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--bpe',
        type=int,
        default=0,
        help='The number of iteration the BPE algorithm did. Default is %(default)s.\
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
    logging.info(strftime('==== text_to_array.py start at %Y%m%d-%H%M$%S ===='))

    if not os.path.isdir(args.corpus_dir_path):
        logging.info('%s does not exist', args.corpus_dir_path)
        exit(1)

    bpe_shapes_list = []
    if args.bpe > 0:
        logging.info('Used BPE: %d', args.bpe)
        with open(to_shape_vocab_file_path(args.corpus_dir_path), 'r', encoding='utf8') as vocabs_file:
            bpe_shapes_list = vocabs_file.read().splitlines()

    npz_path = os.path.join(args.corpus_dir_path, 'arrays.npz')
    if os.path.isfile(to_vocabs_file_path(args.corpus_dir_path)) and os.path.isfile(npz_path):
        if args.use_existed:
            logging.info('Corpus directory: %s already has vocabs file and array file.', args.corpus_dir_path)
            logging.info('Flag --use-existed is set')
            logging.info('==== text_to_array.py exited ====')
            return 0
        else:
            logging.info('Corpus directory: %s already has vocabs file and array file. Remove? (y/n)', args.corpus_dir_path)
            while True:
                i = input()
                if i == 'y':
                    os.remove(to_vocabs_file_path(args.corpus_dir_path))
                    os.remove(npz_path)
                    break
                if i == 'n':
                    logging.info('==== text_to_array.py exited ====')
                    return 0
                print('(y/n):')

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
        existing_file_paths = []
        if os.path.exists(npy_dir_path):
            existing_file_paths.append(npy_dir_path)
        if os.path.exists(npy_zip_path):
            existing_file_paths.append(npy_zip_path)
        if len(existing_file_paths) != 0:
            print(f'Find existing intermidiate file(s): {" ".join(existing_file_paths)}. Removed.')
            shutil.rmtree(npy_dir_path)
            os.remove(npy_zip_path)

        os.makedirs(npy_dir_path)
        for i, p in tqdm(enumerate(corpus_iterator), total=len(corpus_iterator)):
            try:
                array = text_list_to_array(p.split(), vocabs)
            except Exception as e:
                print(f'piece #{i}')
                raise e
            np.save(os.path.join(npy_dir_path, str(i)), array)

        logging.info('Write npys end. time: %.3f', time()-start_time)
        # zip all the npy files into one file with '.npz' extension
        start_time = time()
        logging.info('Begin zipping npys')
        shutil.make_archive(npy_dir_path, 'zip', root_dir=npy_dir_path)
        os.rename(npy_zip_path, npz_path)
        # delete the npys
        shutil.rmtree(npy_dir_path)
        logging.info('Zipping npys end. time: %.3f', time()-start_time)

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
    # with CorpusIterator exit

    start_time = time()
    logging.info('Making statistics')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.pyplot').disabled = True
    from matplotlib import pyplot as plt
    from pandas import Series
    text_stats = {
        'track_number_distribution': Counter(),
        'instrument_distribution': Counter(),
        'token_type_distribution': Counter(),
        'note_number_per_piece' : list(),
        'token_number_per_piece': list()
    }
    with CorpusIterator(args.corpus_dir_path) as corpus_iterator:
        for piece in corpus_iterator:
            text_stats['track_number_distribution'][piece.count(' R')] += 1
            head_end = piece.find(' M') # find first occurence of measure token
            tracks_text = piece[4:head_end]
            track_tokens = tracks_text.split()
            for track_token in track_tokens:
                instrument_id = b36str2int(track_token.split(':')[1])
                text_stats['instrument_distribution'][instrument_id] += 1
            note_token_number = piece.count(' N')
            text_stats['token_type_distribution']['note'] += note_token_number
            text_stats['token_type_distribution']['position'] += piece.count(' P')
            text_stats['token_type_distribution']['tempo'] += piece.count(' T')
            text_stats['token_type_distribution']['measure'] += piece.count(' M')
            text_stats['token_type_distribution']['track'] += piece.count(' R')
            text_stats['note_number_per_piece'].append(note_token_number)
            text_stats['token_number_per_piece'].append(piece.count(" ") + 1)

    text_stats['note_number_per_piece_describe'] = {
        k:float(v) for k, v in dict(Series(text_stats['note_number_per_piece']).describe()).items()
    }
    text_stats['token_number_per_piece_describe'] = {
        k:float(v) for k, v in dict(Series(text_stats['token_number_per_piece']).describe()).items()
    }

    stats_dir_path = os.path.join(args.corpus_dir_path, 'stats')
    if not os.path.exists(stats_dir_path):
        os.mkdir(stats_dir_path)

    # draw graph for each value
    for k, v in text_stats.items():
        plt.figure(figsize=(16.8, 6.4))
        if isinstance(v, Counter):
            plt.title(k)
            if k == 'instrument_distribution':
                instrument_count = [0] * 129
                for p in v:
                    instrument_count[p] = v[p]
                plt.xticks(rotation=90, fontsize='small')
                plt.subplots_adjust(left=0.075, right=1-0.025, bottom=0.25)
                plt.bar(INSTRUMENT_NAMES, instrument_count)
            elif k == 'token_type_distribution':
                plt.barh(list(v.keys()), list(v.values()))
            else:
                plt.bar(list(v.keys()), list(v.values()))
        elif isinstance(v, list):
            # note_number_per_piece and token_number_per_piece
            k_describle = k + '_describe'
            plt.title(k)
            plt.text(
                x=0.01,
                y=0.5,
                s='\n'.join([f'{k}={round(v, 3)}' for k, v in text_stats[k_describle].items()]),
                transform=plt.gcf().transFigure
            )
            plt.subplots_adjust(left=0.125)
            plt.hist(v, 100)
            plt.yscale('log')
        else:
            continue
        plt.savefig(os.path.join(stats_dir_path, f'{k}.png'))
        plt.clf()

    text_stats['track_number_distribution'] = dict(text_stats['track_number_distribution'])
    text_stats['instrument_distribution'] = dict(text_stats['instrument_distribution'])
    text_stats['token_type_distribution'] = dict(text_stats['token_type_distribution'])
    with open(os.path.join(stats_dir_path, 'stats.json'), 'w+', encoding='utf8') as statfile:
        json.dump(text_stats, statfile)
    logging.info('Dumped stats.json at %s', os.path.join(stats_dir_path, 'stats.json'))
    logging.info('Make statistics time: %.3f', time()-start_time)
    
    logging.info('==== text_to_array.py exited ====')
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
