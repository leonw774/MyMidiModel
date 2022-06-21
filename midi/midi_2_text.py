import os
from argparse import ArgumentParser
from multiprocessing import Pool
from time import time

from midi_util import midi_2_text

def mp_worker(args_dict):
    try:
        text = midi_2_text(**args_dict)
    except Exception as e:
        print(args_dict['midi_filepath'])
        print(str(e))
        text = ''
    return text


def mp_handler(
        midi_filepath_set: set,
        mp_work_number: int,
        args_dict: dict):

    error_count = 0
    args_dict_set = list()
    for midi_filepath in midi_filepath_set:
        a = args_dict.copy() # shallow copy of args_dict
        a['midi_filepath'] = midi_filepath
        args_dict_set.append(a)

    print(f'start process with {mp_work_number} workers')

    text_set = set()
    with Pool(mp_work_number) as p:
        for text in p.imap_unordered(mp_worker, args_dict_set):
            if len(text) != 0:
                text_set.add(text)
            else:
                error_count += 1
    print('Error count:', error_count)
    return text_set

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--nth',
        dest='nth',
        type=int,
        default=None,
        help='The time unit length would be the length of a n-th note. Must be multiples of 4. \
            Default is 96.'
    )
    parser.add_argument(
        '--max-track-num',
        dest='max_track_num',
        type=int,
        default=None,
        help='The maximum tracks nubmer to keep in text, if the input midi has more "instruments" than this value, \
            some tracks would be merged or discard. Default is 24.'
    )
    parser.add_argument(
        '--max-duration',
        dest='max_duration',
        type=int,
        default=None,
        help='Max length of duration in unit of quarter note (beat)'
    )
    parser.add_argument(
        '--tempo-quantization',
        nargs=3,
        dest='tempo_quantization',
        type=int,
        default=None,
        metavar=('TEMPO_MIN', 'TEMPO_MAX', 'TEMPO_STEP'),
        help='Three integers: (min, max, step), where min and max are INCLUSIVE. Default is 56, 184, 4'
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
            And it will process all ".mid" files recursively under them.'
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
            All texts from the input midi files will be written into this file, \
            seperated by breaklines. Default is "output.txt".'
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
    print(args_dict)

    filepath_set = set()
    for inpath in args.input_path:
        if args.recursive:
            assert os.path.isdir(inpath)
            for root, dirs, files in os.walk(inpath):
                for filename in files:
                    if filename.endswith('.mid') or filename.endswith('.MID'):
                        filepath_set.add(os.path.join(root, filename))
        else:
            assert os.path.isfile(inpath)
            filepath_set.add(inpath)

    start_time = time()
    text_lengths = []
    if args.mp_work_number <= 1:
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            for filepath in filepath_set:
                text = midi_2_text(filepath, **args_dict)
                text_lengths.append(len(text))
                out_file.write(text + '\n')
    else:
        text_set = mp_handler(filepath_set, args.mp_work_number, args_dict)
        with open(args.output_path, 'w+', encoding='utf8') as out_file:
            for text in text_set:
                out_file.write(text + '\n')
            text_lengths = [len(text) for text in text_set]

    print(f'Processed {len(filepath_set)} files.')
    print(f'Total token count: {sum(text_lengths)}. Average {sum(text_lengths)/len(text_lengths)} per file.')
    print('Output path:', args.output_path)
    print('Time:', time()-start_time)
