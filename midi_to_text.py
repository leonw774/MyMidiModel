import glob
import logging
import os
import shutil
import sys
import zlib
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool
from time import time, strftime
from traceback import format_exc

from tqdm import tqdm

from util import (
    midi_to_text_list,
    to_paras_file_path,
    to_corpus_file_path,
    to_pathlist_file_path,
    dump_corpus_paras,
    CorpusIterator,
    b36str2int
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--nth',
        dest='nth',
        type=int,
        default=96,
        help='The time unit length would be the length of a n-th note. Must be multiples of 4. \
            Default is 96.'
    )
    parser.add_argument(
        '--max-track-number',
        dest='max_track_number',
        type=int,
        default=24,
        help='The maximum tracks nubmer to keep in text, if the input midi has more "instruments" than this value, \
            some tracks would be merged or discard. Default is 24.'
    )
    parser.add_argument(
        '--max-duration',
        dest='max_duration',
        type=int,
        default=4,
        help='Max length of duration in unit of quarter note (beat). Default is 4.'
    )
    parser.add_argument(
        '--velocity-step',
        dest='velocity_step',
        type=int,
        default=16,
        help='Snap the value of velocities to multiple of this number. Default is 16.'
    )
    parser.add_argument(
        '--tempo-quantization',
        nargs=3,
        dest='tempo_quantization',
        type=int,
        default=[120-6*16, 120+5*16, 16], # 24, 200, 16
        metavar=('TEMPO_MIN', 'TEMPO_MAX', 'TEMPO_STEP'),
        help='Three integers: (min, max, step), where min and max are INCLUSIVE. Default is 24, 200, 16.'
    )
    parser.add_argument(
        '--position-method',
        dest='position_method',
        type=str,
        choices=['event', 'attribute'],
        default='attribute',
        help="Could be 'event' or 'attribute'. \
            'attribute' means position info is part of the note token. \
            'event' means position info will be its own event token."
    )
    parser.add_argument(
        '--use-continuing-note',
        dest='use_cont_note',
        action='store_true'
    )
    parser.add_argument(
        '--use-merge-drums',
        dest='use_merge_drums',
        action='store_true'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose'
    )
    parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
        help='Path to the log file. Default is empty, which means no logging would be performed.'
    )
    parser.add_argument(
        '--use-existed',
        dest='use_existed',
        action='store_true',
        help='If the corpus already existed, do nothing.'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        dest='recursive',
        help='If set, the input path will have to be directory path(s). \
            And all ".mid" files under them will be processed recursively.'
    )
    parser.add_argument(
        '-w', '--workers',
        dest='mp_work_number',
        type=int,
        default=1,
        help='The number of worker for multiprocessing. Default is 1. \
            If this number is 1 or below, multiprocessing would not be used.'
    )
    parser.add_argument(
        '-s', '--make-stats',
        action='store_true',
        dest='make_stats',
        help='Record statics of the processed midi files. \
            The output file path is [OUTPUT_PATH] appended by a suffix \"_stat.json\"'
    )
    parser.add_argument(
        '-o', '--output-path',
        dest='output_path',
        type=str,
        default=os.getcwd(),
        help='The path of the directory for the output files. Default is current work directory.'
    )
    parser.add_argument(
        'input_path',
        nargs='+', # store as list and at least one
        help='The path(s) of input files/directodries'
    )
    return parser.parse_args()


def handler(args_dict: dict):
    n = args_dict.pop('n', 0)
    verbose = args_dict.pop('verbose', False)
    logging.debug('%d pid: %d file path: %s', n, os.getpid(), args_dict['midi_file_path'])
    try:
        text_list = midi_to_text_list(**args_dict)
        piece_bytes = ' '.join(text_list).encode()
        # return compressed text because memory usage issue
        return zlib.compress(piece_bytes)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        if verbose:
            logging.debug(format_exc())
        return repr(e)


def loop_func(
        midi_file_path_list,
        out_file,
        args_dict: dict):

    token_number_list = []
    good_path_list = []
    bad_reasons_counter = Counter()
    for n, midi_file_path in tqdm(enumerate(midi_file_path_list), total=len(midi_file_path_list)):
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
        midi_file_path_list: set,
        out_file,
        mp_work_number: int,
        args_dict: dict):

    args_dict_list = list()
    for n, midi_file_path in enumerate(midi_file_path_list):
        a = args_dict.copy() # shallow copy of args_dict
        a['midi_file_path'] = midi_file_path
        a['n'] = n
        args_dict_list.append(a)
    logging.info('Start pool with %d workers', mp_work_number)

    token_number_list = []
    good_path_list = []
    bad_reasons = Counter()
    with Pool(mp_work_number) as p:
        compressed_piece_list = list(tqdm(
            p.imap(handler, args_dict_list),
            total=len(args_dict_list)
        ))
    logging.info('Multi-processing end. Object size: %d bytes', sum(sys.getsizeof(cp) for cp in compressed_piece_list))
    for i, compressed_piece in enumerate(compressed_piece_list):
        if isinstance(compressed_piece, bytes):
            piece = zlib.decompress(compressed_piece).decode()
            out_file.write(piece+'\n')
            token_number_list.append(piece.count(' ')+1) # token number = space number + 1
            good_path_list.append(args_dict_list[i]['midi_file_path'])
        else:
            bad_reasons[compressed_piece] += 1

    logging.info('Bad reasons counter: %s', str(dict(bad_reasons)))
    return token_number_list, good_path_list


def main():
    args = parse_args()
    args_vars = vars(args)
    meta_args = ['input_path', 'output_path', 'make_stats', 'log_file_path', 'mp_work_number', 'recursive', 'use_existed']
    args_dict = {
        k: v
        for k, v in args_vars.items()
        if k not in meta_args and v is not None
    }
    corpus_paras_dict = {
        k: v
        for k, v in args_dict.items()
        if k != 'verbose'
    }

    # when not verbose, only info level or higher will be printed to stdout or stderr and logged into file
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
    logging.info(strftime('==== midi_to_text.py start at %Y%m%d-%H%M ===='))

    args_str = '\n'.join([
        str(k)+':'+str(v) for k, v in args_vars.items()
        if k not in meta_args
    ])
    logging.info(args_str)

    # check if output_path is a directory
    if os.path.exists(args.output_path):
        if (os.path.exists(to_corpus_file_path(args.output_path)) 
                and os.path.exists(to_paras_file_path(args.output_path))
                and os.path.exists(to_pathlist_file_path(args.output_path))):
            if args.use_existed:
                logging.info('Output directory: %s already has corpus file.', args.output_path)
                logging.info('Flag --use-existed is set')
                logging.info('==== midi_to_text.py exited ====')
                return 0
            else:
                shutil.rmtree(args.output_path)
                logging.info('Output directory: %s already has corpus file. Removed.', args.output_path)
        else:
            shutil.rmtree(args.output_path)
            logging.info('Output directory: %s already existed, but no corpus file. Removed.', args.output_path)
    os.makedirs(args.output_path)

    start_time = time()
    file_path_list = list()
    for inpath in args.input_path:
        if args.recursive:
            if not os.path.isdir(inpath):
                print(f'Input path {inpath} is not a directory or doesn\'t exist.')
            file_path_list = glob.glob(inpath+'/**/*.mid', recursive=True)
        else:
            if not os.path.isfile(inpath):
                print(f'Input path {inpath} is not a file or doesn\'t exist.')
            file_path_list.append(inpath)

    if len(file_path_list) == 0:
        logging.info('No file to process')
        logging.info('==== midi_to_text.py exited ====')
        return 1
    else:
        logging.info('Find %d files', len(file_path_list))
    file_path_list.sort()

    # write parameter file
    with open(to_paras_file_path(args.output_path), 'w+', encoding='utf8') as out_paras_file:
        out_paras_file.write(dump_corpus_paras(corpus_paras_dict))

    # write main corpus file
    with open(to_corpus_file_path(args.output_path), 'w+', encoding='utf8') as out_corpus_file:
        if args.mp_work_number <= 1:
            token_number_list, good_path_list = loop_func(file_path_list, out_corpus_file, args_dict)
        else:
            token_number_list, good_path_list = mp_func(file_path_list, out_corpus_file, args.mp_work_number, args_dict)

    logging.info('Processed %d files', len(token_number_list))
    if len(token_number_list) > 0:
        logging.info('Average tokens: %.3f per file', sum(token_number_list)/len(token_number_list))
    else:
        logging.info('No midi file is processed')
        shutil.rmtree(args.output_path)
        logging.info('Removed output directory: %s', args.output_path)
        return 1
    logging.info('Process time: %.3f', time()-start_time)

    with open(to_pathlist_file_path(args.output_path), 'w+', encoding='utf8') as good_path_list_file:
        good_path_list_file.write('\n'.join(good_path_list))
        good_path_list_file.close()

    if args.make_stats:
        start_time = time()
        logging.info('Making statistics')
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.getLogger('matplotlib.pyplot').disabled = True
        import json
        from matplotlib import pyplot as plt
        from pandas import Series
        text_stats = {
            'number_of_pieces': len(good_path_list),
            'track_number_distribution': Counter(),
            'instrument_distribution': Counter(),
            'token_type_distribution': Counter(),
            'note_number_per_piece' : list(),
            'token_number_per_piece': token_number_list
        }
        with CorpusIterator(args.output_path) as corpus_iterator:
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

        text_stats['note_number_per_piece_describe'] = {
            k:float(v) for k, v in dict(Series(text_stats['note_number_per_piece']).describe()).items()
        }
        text_stats['token_number_per_piece_describe'] = {
            k:float(v) for k, v in dict(Series(text_stats['token_number_per_piece']).describe()).items()
        }

        stats_dir_path = os.path.join(args.output_path, 'stats')
        if not os.path.exists(stats_dir_path):
            os.mkdir(stats_dir_path)

        # draw graph for each value
        for k, v in text_stats.items():
            plt.figure(figsize=(12.8, 4.8))
            if isinstance(v, Counter):
                plt.title(k)
                plt.bar(v.keys(), v.values())
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
                plt.subplots_adjust(left=0.15)
                plt.hist(v, 10)
            else:
                continue
            plt.savefig(os.path.join(stats_dir_path, f'{k}.png'))
            plt.clf()

        text_stats['track_number_distribution'] = dict(text_stats['track_number_distribution'])
        text_stats['instrument_distribution'] = dict(text_stats['instrument_distribution'])
        text_stats['token_type_distribution'] = dict(text_stats['token_type_distribution'])
        with open(os.path.join(stats_dir_path, 'stats.json'), 'w+', encoding='utf8') as statfile:
            json.dump(text_stats, statfile)
        logging.info('Wrote stats.json at %s', os.path.join(stats_dir_path, 'stats.json'))
        logging.info('Make statistics time: %.3f', time()-start_time)
    logging.info('==== midi_to_text.py exited ====')


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
