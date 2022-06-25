import os
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool
from time import time
from tqdm import tqdm

from midi_util import midi_2_text, make_para_yaml
from tokens import TokenParseError


def mp_worker(args_dict):
    try:
        return midi_2_text(**args_dict)
    except Exception as e:
        # print(args_dict['midi_filepath'])
        # print(e)
        return None


def mp_handler(
        midi_filepath_set: set,
        mp_work_number: int,
        args_dict: dict):

    args_dict_list = list()
    for midi_filepath in midi_filepath_set:
        a = args_dict.copy() # shallow copy of args_dict
        a['midi_filepath'] = midi_filepath
        args_dict_list.append(a)
    print(f'Start process with {mp_work_number} workers')

    error_count = 0
    text_set = set()
    text_lengths = []
    with Pool(mp_work_number) as p:
        for text in tqdm(p.imap_unordered(mp_worker, args_dict_list)):
            if text is not None:
                text_set.add(text)
                text_lengths.append(len(text))
            else:
                print(len(text_set))
                error_count += 1
        p.close()
        p.join()
    print('Error count:', error_count)
    return text_set, text_lengths


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
        if k not in ['debug', 'use_merge_drums', 'use_merge_sparse']
    }
    print(args_vars)

    filepath_set = set()
    for inpath in args.input_path:
        if args.recursive:
            assert os.path.isdir(inpath), f'Path {inpath} is not a directory or doesn\'t exist.'
            for root, dirs, files in os.walk(inpath):
                for filename in files:
                    if filename.endswith('.mid') or filename.endswith('.MID'):
                        filepath_set.add(os.path.join(root, filename))
        else:
            assert os.path.isfile(inpath), f'Path {inpath} is not a file or doesn\'t exist.'
            filepath_set.add(inpath)
    if len(filepath_set) == 0:
        print('No file to process')
        exit()
    else:
        print(f'Find {len(filepath_set)} files')

    start_time = time()
    text_lengths = []
    if args.mp_work_number <= 1:
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            out_file.write(make_para_yaml(paras_dict))
            for filepath in tqdm(filepath_set):
                try:
                    text = midi_2_text(filepath, **args_dict)
                    text_lengths.append(len(text))
                except Exception as e:
                    print(filepath)
                    print(e)
                    text = ''
                out_file.write(text + '\n')
    else:
        text_set, text_lengths = mp_handler(filepath_set, args.mp_work_number, args_dict)
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            out_file.write(make_para_yaml(paras_dict))
            for text in text_set:
                out_file.write(text + '\n')

    print(f'{len(text_lengths)} files processed')
    print(f'Total token count: {sum(text_lengths)}. Average {sum(text_lengths)/len(text_lengths)} per file')
    print('Output path:', args.output_path)
    print('Time:', time()-start_time)
