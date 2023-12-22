from argparse import ArgumentParser, Namespace
from collections import Counter
import glob
from io import TextIOWrapper
import logging
import os
from multiprocessing import Pool
import shutil
import sys
from time import time, strftime
from traceback import format_exc
import zlib

from miditoolkit import MidiFile
from psutil import cpu_count
from tqdm import tqdm

from util.midi import midi_to_piece
from util.corpus import (
    to_paras_file_path, to_corpus_file_path, to_pathlist_file_path,
    dump_corpus_paras
)


def parse_args():
    handler_args_parser = ArgumentParser()
    handler_args_parser.add_argument(
        '--tpq',
        type=int,
        default=12,
        help='The time unit length would be the length of a n-th note. \
            Must be a even number. Default is %(default)s.'
    )
    handler_args_parser.add_argument(
        '--max-track-number',
        type=int,
        default=24,
        help='The maximum tracks nubmer to keep in text, \
            if the input midi has more "instruments" than this value, \
            some tracks would be merged or discard. \
            Default is %(default)s.'
    )
    handler_args_parser.add_argument(
        '--max-duration',
        type=int,
        default=48,
        help='Max duration value in unit of ticks. Default is %(default)s.'
    )
    handler_args_parser.add_argument(
        '--velocity-step',
        type=int,
        default=16,
        help='Snap the value of velocities to multiples of this number. \
            Default is %(default)s.'
    )
    handler_args_parser.add_argument(
        '--tempo-quantization',
        nargs=3,
        type=int,
        default=[120-6*16, 120+5*16, 16], # 24, 200, 16
        metavar=('TEMPO_MIN', 'TEMPO_MAX', 'TEMPO_STEP'),
        help='Three integers: (min, max, step). \
            The min and max are INCLUSIVE. \
            Default is %(default)s.'
    )
    handler_args_parser.add_argument(
        '--use-continuing-note',
        dest='use_cont_note',
        action='store_true'
    )

    main_parser = ArgumentParser()
    main_parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose'
    )
    main_parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
        help='Path to the log file. \
            Default is empty, which means no logging would be performed.'
    )
    main_parser.add_argument(
        '--use-existed',
        action='store_true',
        help='If the corpus already existed, do nothing.'
    )
    main_parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='If set, the input path will have to be directory path(s). \
            And all ".mid" files under them will be processed recursively.'
    )
    main_parser.add_argument(
        '--worker-number', '-w',
        type=int,
        default=1,
        help='The number of worker for multiprocessing. \
            Default is %(default)s. \
            If this number is 1 or below, multiprocessing would not be used.'
    )
    main_parser.add_argument(
        '--output-dir-path', '-o',
        type=str,
        default='midi_to_corpus_output',
        help='The path of the directory for the corpus.'
    )
    main_parser.add_argument(
        'input_path_list',
        type=str,
        nargs='+', # store as list and at least one
        help='The path(s) of input files/directodries'
    )
    args = dict()
    args['handler_args'], others = handler_args_parser.parse_known_args()
    args.update(vars(
        main_parser.parse_known_args(others)[0]
    ))
    return Namespace(**args)


def handler(args_dict: dict):
    n = args_dict.pop('n', 0)
    midi_file_path  = args_dict.pop('midi_file_path', '')
    try:
        args_dict['midi'] = MidiFile(midi_file_path)
    except Exception:
        logging.debug(
            '\n%d pid: %d file: %s\n%s',
            n, os.getpid(), midi_file_path, format_exc()
        )
        return 'RuntimeError(\'miditoolkit failed to parse the file\')'

    try:
        piece = midi_to_piece(**args_dict)
        piece_bytes = piece.encode()
        # return compressed text because memory issue
        return zlib.compress(piece_bytes)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        # if not (
        #         repr(e).startswith('Assert') or repr(e).startswith('Runtime')
        #     ):
        logging.debug(
            '\n%d pid: %d file: %s\n%s',
            n, os.getpid(), midi_file_path, format_exc()
        )
        return repr(e)


def loop_func(
        midi_file_path_list: list,
        out_file: TextIOWrapper,
        args_dict: dict):

    token_number_list = []
    good_path_list = []
    bad_reasons_counter = Counter()
    tqdm_enum_midi_file_path_list = tqdm(
        enumerate(midi_file_path_list),
        total=len(midi_file_path_list),
        ncols=80
    )
    for n, midi_file_path in tqdm_enum_midi_file_path_list:
        n_args_dict = dict(args_dict)
        n_args_dict['n'] = n
        n_args_dict['midi_file_path'] = midi_file_path
        compressed_piece = handler(n_args_dict)
        if isinstance(compressed_piece, bytes):
            piece = zlib.decompress(compressed_piece).decode()
            out_file.write(piece + '\n')
            token_number_list.append(piece.count(' ')+1)
            good_path_list.append(midi_file_path)
        else:
            bad_reasons_counter[compressed_piece] += 1
    logging.info('Bad reasons counter: %s', str(dict(bad_reasons_counter)))
    return token_number_list, good_path_list


def mp_func(
        midi_file_path_list: list,
        out_file: TextIOWrapper,
        worker_number: int,
        args_dict: dict):

    args_dict_list = list()
    for n, midi_file_path in enumerate(midi_file_path_list):
        a = args_dict.copy() # shallow copy of args_dict
        a['midi_file_path'] = midi_file_path
        a['n'] = n
        args_dict_list.append(a)
    logging.info('Start pool with %d workers', worker_number)

    token_number_list = []
    good_path_list = []
    bad_reasons = Counter()
    with Pool(worker_number) as p:
        compressed_piece_list = list(tqdm(
            p.imap(handler, args_dict_list),
            total=len(args_dict_list),
            ncols=80
        ))
    logging.info(
        'Multi-processing end. Object size: %d bytes',
        sum(sys.getsizeof(cp) for cp in compressed_piece_list)
    )
    for i, compressed_piece in enumerate(compressed_piece_list):
        if isinstance(compressed_piece, bytes):
            piece = zlib.decompress(compressed_piece).decode()
            out_file.write(piece+'\n')
            # token number = space number + 1
            token_number_list.append(piece.count(' ') + 1)
            good_path_list.append(args_dict_list[i]['midi_file_path'])
        else:
            bad_reasons[compressed_piece] += 1

    logging.info('Bad reasons counter: %s', str(dict(bad_reasons)))
    return token_number_list, good_path_list


def main():
    args = parse_args()
    corpus_paras_dict = dict(vars(args.handler_args)) # a dict() for copy

    # when not verbose, only info level or higher will be printed to stdout
    # and also logged into file
    loglevel = logging.DEBUG if args.verbose else logging.INFO
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
    logging.info(
        strftime('==== midi_to_corpus.py start at %Y%m%d-%H%M%SS ====')
    )

    logging.info(
        '\n'.join([
            f'{k}:{v}' for k, v in vars(args).items()
            if k != 'handler_args'
        ])
    )
    logging.info(
        '\n'.join([
            f'{k}:{v}' for k, v in corpus_paras_dict.items()
        ])
    )

    corpus_file_path = to_corpus_file_path(args.output_dir_path)
    if os.path.isdir(args.output_dir_path):
        if os.path.exists(corpus_file_path):
            logging.info('Output corpus: %s already exist.', corpus_file_path)
            if args.use_existed:
                logging.info('Flag --use-existed is set')
                logging.info('==== midi_to_corpus.py exited ====')
                return 0

            logging.info('Remove?')
            reply_str = ''
            while reply_str not in ('y', 'n'):
                reply_str = input('(y=remove/n=exit):').lower()

            if reply_str == 'n':
                logging.info('==== midi_to_corpus.py exited ====')
                return 0

            os.remove(to_corpus_file_path(args.output_dir_path))
            paras_path = to_paras_file_path(args.output_dir_path)
            if os.path.exists(paras_path):
                os.remove(paras_path)
            pathlist_path = to_pathlist_file_path(args.output_dir_path)
            if os.path.exists(pathlist_path):
                os.remove(pathlist_path)
            print('Removed')

        else:
            logging.info(
                'Output directory: %s already existed, but no corpus files.',
                args.output_dir_path
            )
    elif os.path.isfile(args.output_dir_path):
        logging.info(
            'Output directory path: %s is a file.',
            args.output_dir_path
        )
        return 1
    else:
        os.makedirs(args.output_dir_path)

    start_time = time()
    file_path_list = list()
    for input_path in args.input_path_list:
        logging.info('Globbing %s', input_path)
        if args.recursive:
            if not os.path.isdir(input_path):
                print(
                    f'Input path {input_path} is not a directory'
                    f' or does not exist.'
                )
            file_path_list = (
                glob.glob(input_path+'/**/*.mid', recursive=True)
                + glob.glob(input_path+'/**/*.midi', recursive=True)
                + glob.glob(input_path+'/**/*.MID', recursive=True)
            )
        else:
            if not os.path.isfile(input_path):
                print(
                    f'Input path {input_path} is not a file'
                    f' or does not exist.'
                )
            file_path_list.append(input_path)

    if len(file_path_list) == 0:
        logging.info('No file to process')
        logging.info('==== midi_to_corpus.py exited ====')
        return 1
    else:
        logging.info('Find %d files', len(file_path_list))
    file_path_list.sort()

    # write parameter file
    paras_file_path = to_paras_file_path(args.output_dir_path)
    with open(paras_file_path, 'w+', encoding='utf8') as out_paras_file:
        out_paras_file.write(dump_corpus_paras(corpus_paras_dict))

    # write main corpus file
    with open(corpus_file_path, 'w+', encoding='utf8') as out_corpus_file:
        handler_args_dict = vars(args.handler_args)
        if args.worker_number <= 1:
            token_number_list, good_path_list = loop_func(
                midi_file_path_list=file_path_list,
                out_file=out_corpus_file,
                args_dict=handler_args_dict
            )
        else:
            worker_number = min(cpu_count(), args.worker_number)
            token_number_list, good_path_list = mp_func(
                midi_file_path_list=file_path_list,
                out_file=out_corpus_file,
                worker_number=worker_number,
                args_dict=handler_args_dict
            )

    logging.info('Processed %d files', len(token_number_list))
    if len(token_number_list) > 0:
        logging.info(
            'Avg. number tokens: %.6f per file',
            sum(token_number_list) / len(token_number_list)
        )
    else:
        logging.info('No midi file is processed')
        shutil.rmtree(args.output_dir_path)
        logging.info('Removed output directory: %s', args.output_dir_path)
        return 1
    logging.info('Process time: %.6f', time()-start_time)

    pathlist_file_path = to_pathlist_file_path(args.output_dir_path)
    with open(pathlist_file_path, 'w+', encoding='utf8') as good_path_list_file:
        good_path_list_file.write('\n'.join(good_path_list))
        good_path_list_file.close()

    logging.info('==== midi_to_corpus.py exited ====')
    return 0


if __name__ == '__main__':
    EXIT_CODE = main()
    exit(EXIT_CODE)
