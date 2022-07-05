import os
import sys
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool
from time import time
from tqdm import tqdm

from midi_util import midi_to_text_list, make_para_yaml
from tokens import TokenParseError


def mp_worker(args_dict: dict):
    n = args_dict.pop('n', 0)
    verbose = args_dict.pop('verbose', False)
    if verbose:
        print(n, 'pid', os.getpid(), args_dict['midi_filepath'])
    try:
        text_list = midi_to_text_list(**args_dict)
        # print(n, len(text_list))
        return text_list
    except Exception as e:
        if verbose:
            print(n, traceback.format_exc())
        return []


def mp_handler(
        midi_filepath_list: set,
        mp_work_number: int,
        args_dict: dict):

    args_dict_list = list()
    for n, midi_filepath in enumerate(midi_filepath_list):
        a = args_dict.copy() # shallow copy of args_dict
        a['midi_filepath'] = midi_filepath
        a['n'] = n
        args_dict_list.append(a)
    print(f'Start process with {mp_work_number} workers')

    exception_count = 0
    good_filepath_list = []
    good_text_lists = []
    with Pool(mp_work_number) as p:
        text_list_list = list(tqdm(
            p.imap(mp_worker, args_dict_list),
            total=len(args_dict_list)
        ))
    print('mp end. object size:', sys.getsizeof(text_list_list))
    for i, text_list in enumerate(text_list_list):
        if len(text_list) != 0:
            good_text_lists.append(text_list)
            good_filepath_list.append(midi_filepath_list[i])
        else:
            exception_count += 1
    text_lists_lengths = list(map(len, good_text_lists))
    print('Bad file count:', exception_count)
    return good_text_lists, text_lists_lengths, good_filepath_list


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
        default=32,
        help='The maximum tracks nubmer to keep in text, if the input midi has more "instruments" than this value, \
            some tracks would be merged or discard. Default is 32.'
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
        default=(120-4*28, 120+4*36, 4), # 8, 264, 4
        metavar=('TEMPO_MIN', 'TEMPO_MAX', 'TEMPO_STEP'),
        help='Three integers: (min, max, step), where min and max are INCLUSIVE. Default is 8, 264, 4'
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
        if k not in ['input_path', 'output_path', 'mp_work_number', 'recursive'] and v is not None
    }
    paras_dict = {
        k: v
        for k, v in args_dict.items()
        if k not in ['debug', 'verbose', 'use_merge_drums', 'use_merge_sparse']
    }
    print(args_vars)

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
        print('No file to process')
        exit()
    else:
        print(f'Find {len(filepath_list)} files')

    start_time = time()
    if args.mp_work_number <= 1:
        text_lists_lengths = []
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            out_file.write(make_para_yaml(paras_dict))
            verbose = args_dict.pop('verbose', False)
            for n, filepath in tqdm(enumerate(filepath_list), total=len(filepath_list)):
                if verbose:
                    print(n, filepath)
                try:
                    text_list = midi_to_text_list(filepath, **args_dict)
                except Exception as e:
                    if verbose:
                        print(traceback.format_exc())
                    text_list = []
                if len(text_list) > 0:
                    text_lists_lengths.append(len(text_list))
                    out_file.write(' '.join(text_list) + '\n')
    else:
        good_text_lists, text_lists_lengths, good_filepath_list = mp_handler(filepath_list, args.mp_work_number, args_dict)
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            out_file.write(make_para_yaml(paras_dict))
            for text_list in good_text_lists:
                out_file.write(' '.join(text_list) + '\n')

    print(f'{len(text_lists_lengths)} files processed')
    if len(text_lists_lengths) > 0:
        print(f'Average tokens: {sum(text_lists_lengths)/len(text_lists_lengths)} per file')
    print('Output path:', args.output_path)
    print('Time:', time()-start_time)
