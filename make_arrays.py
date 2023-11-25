from argparse import ArgumentParser
from collections import Counter
import json
import logging
from multiprocessing import Pool
import os
import shutil
from time import strftime, time
from traceback import format_exc
from typing import Dict
from zipfile import ZipFile, ZIP_DEFLATED

from matplotlib import pyplot as plt
import numpy as np
from pandas import Series
from tqdm import tqdm

from util.tokens import (
    INSTRUMENT_NAMES,
    SEP_TOKEN_STR,
    TRACK_EVENTS_CHAR, MEASURE_EVENTS_CHAR, TEMPO_EVENTS_CHAR,
    POSITION_EVENTS_CHAR, MULTI_NOTE_EVENTS_CHAR, NOTE_EVENTS_CHAR
)
from util.tokens import b36strtoi
from util.vocabs import build_vocabs
from util.corpus_reader import CorpusReader
from util.corpus import (
    to_shape_vocab_file_path, to_vocabs_file_path, to_arrays_file_path,
    get_corpus_paras, text_list_to_array, get_full_array_string
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--bpe',
        action='store_true',
        help='If set, it means that BPE was performed, \
            and we will try to read the shape_vocab file under corpus_dir_path.'
    )
    parser.add_argument(
        '--worker-number',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
        help='Path to the log file. Default is empty, \
            which means no logging would be performed.'
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


def mp_handler(args_dict):
    i = args_dict['i']
    args_dict.pop('i')
    try:
        a = text_list_to_array(**args_dict)
        return a
    except Exception:
        print(f'Error at piece #{i}')
        print(format_exc())
        return np.empty(shape=(0,), dtype=np.uint16)


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
    logging.info(strftime('==== make_arrays.py start at %Y%m%d-%H%M%S ===='))

    if not os.path.isdir(args.corpus_dir_path):
        logging.info('%s does not exist', args.corpus_dir_path)
        exit(1)

    bpe_shapes_list = []
    if args.bpe:
        logging.info('Used BPE: True')
        shape_vocab_file_path = to_shape_vocab_file_path(args.corpus_dir_path)
        with open(shape_vocab_file_path, 'r', encoding='utf8') as vocabs_file:
            bpe_shapes_list = vocabs_file.read().splitlines()

    vocab_path = to_vocabs_file_path(args.corpus_dir_path)
    npz_path = to_arrays_file_path(args.corpus_dir_path)
    if os.path.isfile(vocab_path) and os.path.isfile(npz_path):
        logging.info(
            'Corpus directory: %s already has vocabs file and arrays file.',
            args.corpus_dir_path
        )
        if args.use_existed:
            logging.info('Flag --use-existed is set')
            logging.info('==== make_arrays.py exited ====')
            return 0

        logging.info("Remove?")
        reply_str = ""
        while reply_str not in ('y', 'n'):
            reply_str = input('(y=remove/n=exit):').lower()

        if reply_str == 'n':
            logging.info('==== make_arrays.py exited ====')
            return 0

        os.remove(vocab_path)
        os.remove(npz_path)
        logging.info('Removed %s and %s', npz_path, vocab_path)

    logging.info('Begin build vocabs for %s', args.corpus_dir_path)
    corpus_paras = get_corpus_paras(args.corpus_dir_path)
    with CorpusReader(args.corpus_dir_path) as corpus_reader:
        assert len(corpus_reader) > 0, f'empty corpus: {args.corpus_dir_path}'

        vocabs, summary_string = build_vocabs(corpus_reader, corpus_paras, bpe_shapes_list)
        logging.info(summary_string)

        with open(vocab_path, 'w+', encoding='utf8') as vocabs_file:
            json.dump(vocabs.to_dict(), vocabs_file)

        start_time = time()
        logging.info('Begin make npys')

        # handle existed files/dirs
        npy_dir_path = os.path.join(args.corpus_dir_path, 'arrays')
        npy_zip_path = os.path.join(args.corpus_dir_path, 'arrays.zip')
        existing_file_paths = []
        if os.path.exists(npy_dir_path):
            existing_file_paths.append(npy_dir_path)
            shutil.rmtree(npy_dir_path)
        if os.path.exists(npy_zip_path):
            existing_file_paths.append(npy_zip_path)
            os.remove(npy_zip_path)
        if len(existing_file_paths) != 0:
            print(
                f'Find existing intermidiate file(s):'\
                f'{" ".join(existing_file_paths)}. Removed.'
            )

        os.makedirs(npy_dir_path)
        args_list = [
            {'text_list': p.split(), 'vocabs': vocabs, 'i': i}
            for i, p in enumerate(corpus_reader)
        ]
        array_list = []
        with Pool(args.worker_number) as p:
            tqdm_text_list_to_array = tqdm(
                p.imap(mp_handler, args_list),
                total=len(corpus_reader),
                ncols=100
            )
            array_list = list(tqdm_text_list_to_array)

        for i, array in enumerate(array_list):
            if array.size != 0:
                np.save(os.path.join(npy_dir_path, str(i)), array)

        logging.info('Make npys end. time: %.3f', time()-start_time)
        # zip all the npy files into one file with '.npz' extension
        start_time = time()
        logging.info('Zipping npys')
        zipfile_cm = ZipFile(npy_zip_path, 'x', compression=ZIP_DEFLATED, compresslevel=1)
        with zipfile_cm as npz_file:
            for file_name in tqdm(os.listdir(npy_dir_path), ncols=100):
                # arcname is file_name -> file should be at root
                npz_file.write(os.path.join(npy_dir_path, file_name), file_name)
        os.rename(npy_zip_path, npz_path)
        # delete the npys
        shutil.rmtree(npy_dir_path)
        logging.info('Zipping npys end. time: %.3f', time()-start_time)

        # for debugging
        if args.debug:
            debug_txt_path = os.path.join(args.corpus_dir_path, 'make_array_debug.txt')
            print(f'Write debug file: {debug_txt_path}')

            piece_0 = next(iter(corpus_reader))
            original_text_list = piece_0.split()
            original_text_list = ['original_text'] + original_text_list
            longest_text_length = max(map(len, original_text_list))
            original_text_list = [
                text.ljust(longest_text_length)
                for text in original_text_list
            ]

            array_data = text_list_to_array(piece_0.split(), vocabs)
            debug_str = get_full_array_string(array_data, vocabs)
            debug_str_list = debug_str.splitlines()

            merged_lines = [
                f'{origi} {debug}'
                for origi, debug in zip(original_text_list, debug_str_list)
            ]
            with open(debug_txt_path, 'w+', encoding='utf8') as f:
                f.write('\n'.join(merged_lines))

        start_time = time()
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.getLogger('matplotlib.pyplot').disabled = True
        token_types = [
            'multinote', 'tempo', 'position', 'measure', 'track', 'all'
        ]

        distributions = {
            'instrument': Counter(),
            'shape': Counter(),
            'tokens_number': {
                t: [] for t in token_types
            }
        }
        regular_single_note_str = ' ' + NOTE_EVENTS_CHAR + ':'
        cont_single_note_str = ' ' + NOTE_EVENTS_CHAR + '~:'
        for piece in tqdm(corpus_reader, desc='Making statistics', ncols=100):
            if args.bpe:
                distributions['shape']['0,0,1;'] += piece.count(regular_single_note_str)
                distributions['shape']['0,0,1~;'] += piece.count(cont_single_note_str)
                for text in piece.split():
                    if text[0] == MULTI_NOTE_EVENTS_CHAR:
                        distributions['shape'][text[1:].split(':')[0]] += 1

            head_end = piece.find(SEP_TOKEN_STR) # find separator
            tracks_text = piece[4:head_end]
            track_tokens = tracks_text.split()
            for track_token in track_tokens:
                instrument_id = b36strtoi(track_token.split(':')[0][1:])
                distributions['instrument'][instrument_id] += 1

            multinote_number = piece.count(' '+NOTE_EVENTS_CHAR)
            multinote_number += piece.count(' '+MULTI_NOTE_EVENTS_CHAR)
            tempo_number = piece.count(' '+TEMPO_EVENTS_CHAR)
            position_number = piece.count(' '+POSITION_EVENTS_CHAR)
            measure_number = piece.count(' '+MEASURE_EVENTS_CHAR)
            track_number = piece.count(' '+TRACK_EVENTS_CHAR)
            all_token_number = piece.count(' ') + 1
            numbers = [
                multinote_number,
                tempo_number,
                position_number,
                measure_number,
                track_number,
                all_token_number
            ]
            for i, token_type in enumerate(token_types):
                distributions['tokens_number'][token_type].append(numbers[i])
    # with CorpuesReader exit

    descriptions = dict()
    for token_type, numbers_per_piece in distributions['tokens_number'].items():
        numbers_per_piece: list
        d: Dict[str, np.float64] = dict(Series(numbers_per_piece).dropna().describe())
        descriptions[token_type] = {
            k: float(v) for k, v in d.items()
        }

    stats_dir_path = os.path.join(args.corpus_dir_path, 'stats')
    if not os.path.exists(stats_dir_path):
        os.mkdir(stats_dir_path)

    # draw graph for each value
    plt.figure(figsize=(16.8, 6.4))
    plt.title('instrument distribution')
    plt.xticks(rotation=90, fontsize='small')
    plt.subplots_adjust(left=0.075, right=1-0.025, bottom=0.25)
    instrument_count = [0] * 129
    for program, count in distributions['instrument'].items():
        instrument_count[program] = count
    plt.bar(INSTRUMENT_NAMES, instrument_count)
    plt.savefig(os.path.join(stats_dir_path, 'instrument_distribution.png'))
    plt.clf()

    if args.bpe:
        sorted_nondefault_shape_count = sorted(
            [
                (count, shape)
                for shape, count in distributions['shape'].items()
                if len(shape) > 7
            ],
            # "if len(shape) > 7" is to remove single note shapes
            reverse=True
        )
        total_shape_num = sum([v for v in distributions['shape'].values()])
        plt.figure(figsize=(16.8, 7.2))
        plt.title('bpe learned shapes distribution')
        plt.xticks(rotation=90, fontsize='small')
        plt.ylabel('Appearance frequency in corpus')
        plt.subplots_adjust(left=0.075, right=1-0.025, bottom=0.25)
        plt.bar(
            x=[
                s[:25]+'...' if len(s) > 25 else s
                for _, s in sorted_nondefault_shape_count
            ],
            height=[c / total_shape_num for c, _ in sorted_nondefault_shape_count],
        )
        plt.savefig(os.path.join(stats_dir_path, 'nondefault_shape_distribution.png'))
        plt.clf()

    plt.figure(figsize=(16.8, 6.4))
    plt.title('track_number distribution')
    plt.xticks(rotation=90, fontsize='small')
    plt.text(
        x=0.01,
        y=0.5,
        s='\n'.join([f'{e}={round(v, 3)}' for e, v in descriptions['track'].items()]),
        transform=plt.gcf().transFigure
    )
    plt.subplots_adjust(left=0.125)
    max_track_count = max(distributions['tokens_number']['track'])
    plt.hist(distributions['tokens_number']['track'], bins=max_track_count)
    plt.xlabel('number of tracks')
    plt.ylabel('number of pieces')
    plt.savefig(os.path.join(stats_dir_path, 'number_of_track_distribution.png'))
    plt.clf()

    plt.figure(figsize=(16.8, 6.4))
    plt.title('number of note/multinote tokens')
    plt.text(
        x=0.01,
        y=0.5,
        s='\n'.join([f'{e}={round(v, 3)}' for e, v in descriptions['multinote'].items()]),
        transform=plt.gcf().transFigure
    )
    plt.subplots_adjust(left=0.125)
    multinote_numbers_per_piece = distributions['tokens_number']['multinote']
    plt.hist(multinote_numbers_per_piece, bins=min(100, len(multinote_numbers_per_piece)))
    plt.yscale('log')
    plt.xlabel('number of note/multinote tokens')
    plt.ylabel('number of pieces')
    plt.savefig(os.path.join(stats_dir_path, 'number_of_multinote_tokens_distribution.png'))
    plt.clf()

    plt.figure(figsize=(16.8, 6.4))
    plt.title('token types distribution')
    token_number_sum = [
        sum(distributions['tokens_number'][t])
        for t in token_types[:-1] # not use 'all'
    ]
    plt.barh(token_types[:-1], token_number_sum)
    plt.savefig(os.path.join(stats_dir_path, 'token_types_distribution.png'))
    plt.clf()

    stats_dict = {
        'distribution': distributions,
        'descriptions': descriptions
    }
    with open(os.path.join(stats_dir_path, 'stats.json'), 'w+', encoding='utf8') as statfile:
        json.dump(stats_dict, statfile)
    logging.info('Dumped stats.json at %s', os.path.join(stats_dir_path, 'stats.json'))
    logging.info('Make statistics time: %.3f', time()-start_time)

    logging.info('==== make_arrays.py exited ====')
    return 0

if __name__ == '__main__':
    EXIT_CODE = main()
    exit(EXIT_CODE)
