import logging
import os
import sys
import traceback
import zlib
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool
from time import time, localtime, strftime
from tqdm import tqdm

from midi_util import midi_to_text_list, make_para_yaml, file_to_paras_and_pieces_iterator
from tokens import TokenParseError, tokenstr2int


def mp_worker(args_dict: dict):
    n = args_dict.pop('n', 0)
    logging.debug('%d pid: %d filepath: %s', n, os.getpid(), args_dict['midi_filepath'])
    try:
        text_list = midi_to_text_list(**args_dict)
        # return compressed text because memory usage issue
        piece = ' '.join(text_list)
        return zlib.compress(piece.encode())
    except:
        logging.debug('%d %s', n, traceback.format_exc())
        return b''


def mp_handler(
        midi_filepath_list: set,
        output_path: str,
        mp_work_number: int,
        args_dict: dict):

    args_dict_list = list()
    for n, midi_filepath in enumerate(midi_filepath_list):
        a = args_dict.copy() # shallow copy of args_dict
        a['midi_filepath'] = midi_filepath
        a['n'] = n
        args_dict_list.append(a)
    logging.info('Start process with %d workers', mp_work_number)

    bad_count = 0
    good_filepath_list = []
    token_number_list = []
    with Pool(mp_work_number) as p:
        compressed_piece_list = list(tqdm(
            p.imap(mp_worker, args_dict_list),
            total=len(args_dict_list)
        ))
    logging.info('mp end. object size: %d bytes', sum(sys.getsizeof(cp) for cp in compressed_piece_list))
    with open(output_path, 'w', encoding='utf8') as outfile:
        for i, compressed_piece in enumerate(compressed_piece_list):
            if len(compressed_piece) > 0:
                piece = zlib.decompress(compressed_piece).decode()
                outfile.write(piece+'\n')
                token_number_list.append(piece.count(' ')+1) # token number = space number + 1
                good_filepath_list.append(midi_filepath_list[i])
            else:
                bad_count += 1
    logging.info('Bad file: %d', bad_count)
    return token_number_list, good_filepath_list


if __name__ == '__main__':
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
        default=8,
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
        default=(120-16*6, 120+16*5, 4), # 24, 200, 16
        metavar=('TEMPO_MIN', 'TEMPO_MAX', 'TEMPO_STEP'),
        help='Three integers: (min, max, step), where min and max are INCLUSIVE. Default is 24, 200, 16'
    )
    parser.add_argument(
        '--tempo-method',
        dest='tempo_method',
        type=str,
        choices=['measure_attribute', 'position_attribute', 'measure_event', 'position_event'],
        default='position_event',
        help='Could be one of four options:\
            \'measure_attribute\', \'position_attribute\', \'measure_event\' and \'position_event\'. \
            \'attribute\' means tempo info is part of the measure or position event token \
            \'event\' means tempo will be its own token, and it will be in the sequence where there tempo change occurs. \
            The token will be placed after the measure token and before position token'
    )
    parser.add_argument(
        '--not-merge-drums',
        action='store_false',
        dest='use_merge_drums'
    )
    parser.add_argument(
        '--not-merge-sparse',
        action='store_false',
        dest='use_merge_sparse'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        dest='debug'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose'
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
        help='Record statics of the processed midi files. The output file path is [OUTPUT_PATH] appended by a suffix \"_stat.json\"'
    )
    parser.add_argument(
        '-o', '--output-path',
        dest='output_path',
        type=str,
        default=os.path.join(os.getcwd(), 'output.txt'),
        help='The path of the output file of the inputs files/directories. \
            All output texts will be written into this file, seperated by breaklines. Default is "output.txt".'
    )
    parser.add_argument(
        'input_path',
        nargs='+', # store as list and at least one
        help='The path(s) of input files/dictionaries'
    )

    args = parser.parse_args()
    args_vars = vars(args)
    args_dict = {
        k: v
        for k, v in args_vars.items()
        if k not in ['input_path', 'output_path', 'make_stats', 'mp_work_number', 'recursive', 'verbose'] and v is not None
    }
    paras_dict = {
        k: v
        for k, v in args_dict.items()
        if k not in ['debug', 'verbose', 'use_merge_drums', 'use_merge_sparse']
    }

    # when not verbose, only info level or higher will be printed to stderr and logged into file
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        filename=strftime("logs/midi_to_text-%Y%m%d-%H%M.log", localtime()),
        filemode='w+',
        level=loglevel
    )
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    logging.getLogger().addHandler(console)

    args_str = '\n'.join([
        str(k)+':'+str(v) for k, v in args_vars.items()
    ])
    logging.info(args_str)

    start_time = time()
    filepath_list = list()
    for inpath in args.input_path:
        if args.recursive:
            assert os.path.isdir(inpath), f'Path {inpath} is not a directory or doesn\'t exist.'
            for root, dirs, files in os.walk(inpath):
                for filename in files:
                    if filename.endswith('.mid') or filename.endswith('.MID'):
                        filepath_list.append(os.path.join(root, filename))
        else:
            assert os.path.isfile(inpath), f'Path {inpath} is not a file or doesn\'t exist.'
            filepath_list.append(inpath)
    filepath_list.sort()
    if len(filepath_list) == 0:
        logging.info('No file to process')
        exit(1)
    else:
        logging.info('Find %d files', len(filepath_list))

    if args.mp_work_number <= 1:
        token_number_list = []
        good_filepath_list = []
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            out_file.write(make_para_yaml(paras_dict))
            for n, filepath in tqdm(enumerate(filepath_list), total=len(filepath_list)):
                logging.debug('%d %s', n, filepath)
                try:
                    text_list = midi_to_text_list(filepath, **args_dict)
                except:
                    logging.debug(traceback.format_exc())
                    text_list = []
                if len(text_list) > 0:
                    token_number_list.append(len(text_list))
                    good_filepath_list.append(filepath)
                    out_file.write(' '.join(text_list) + '\n')
    else:
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            out_file.write(make_para_yaml(paras_dict))
        token_number_list, good_filepath_list = mp_handler(filepath_list, args.output_path, args.mp_work_number, args_dict)

    logging.info('Processed %d files', len(token_number_list))
    if len(token_number_list) > 0:
        logging.info('Average tokens: %.3f per file', sum(token_number_list)/len(token_number_list))
    logging.info('Process time: %.3f', time()-start_time)

    if args.make_stats:
        start_time = time()
        logging.info('Making statistics')
        import json
        from pandas import Series
        text_stats = {
            'track_number_distribution': Counter(),
            'instrument_distribution': Counter(),
            'token_type_distribution': Counter(),
            'note_number_per_piece' : list(),
        }
        with open(args.output_path, 'r', encoding='utf8') as outfile:
            paras, piece_iterator = file_to_paras_and_pieces_iterator(outfile)
            for piece in piece_iterator:
                text_stats['track_number_distribution'][piece.count(' R')] += 1
                head_end = piece.find(' M') # find first occurence of measure token
                tracks_text = piece[4:head_end]
                track_tokens = tracks_text.split()
                for track_token in track_tokens:
                    instrument_id = tokenstr2int(track_token.split(':')[1])
                    text_stats['instrument_distribution'][instrument_id] += 1
                note_token_number = piece.count(' N')
                text_stats['token_type_distribution']['note'] += note_token_number
                text_stats['token_type_distribution']['position'] += piece.count(' P')
                text_stats['token_type_distribution']['tempo'] += piece.count(' T')
                text_stats['token_type_distribution']['measure'] += piece.count(' M')
                text_stats['token_type_distribution']['track'] += piece.count(' R')
                text_stats['note_number_per_piece'].append(note_token_number)

        text_stats['track_number_distribution'] = dict(text_stats['track_number_distribution'])
        text_stats['instrument_distribution'] = dict(text_stats['instrument_distribution'])
        text_stats['token_type_distribution'] = dict(text_stats['token_type_distribution'])
        text_stats['note_number_per_piece_describe'] = {
            k:float(v) for k, v in dict(Series(text_stats['note_number_per_piece']).describe()).items()
        }
        del text_stats['note_number_per_piece']

        with open(args.output_path+'_stat.json', 'w+', encoding='utf8') as statfile:
            statfile.write(json.dumps(text_stats))
        logging.info('Write stats file to %s', args.output_path+'_stat.json')
        logging.info('Stat time: %.3f', time()-start_time)
    logging.info('midi_to_text.py exited')
