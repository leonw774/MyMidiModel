from argparse import ArgumentParser
import glob
import json
import logging
from multiprocessing import Pool
import os
from time import strftime, time
from traceback import format_exc
from typing import List, Dict
import random

from miditoolkit import MidiFile
import numpy as np
from pandas import Series
from tqdm import tqdm

from util.evaluations import EVAL_FEATURE_NAMES, midi_to_features

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--log',
        dest='log_file_path',
        type=str,
        default='',
    )
    parser.add_argument(
        '--sample-number',
        type=int,
        default=100
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=20
    )
    parser.add_argument(
        '--max-pairs-number',
        type=int,
        default=int(1e6)
    )
    parser.add_argument(
        'dataset_dir_path',
        type=str
    )
    return parser.parse_args()


def midi_to_features_wrapper(args_dict: dict):
    try:
        midifile_obj = MidiFile(args_dict['midi_path'])
    except Exception as e:
        return None
        midifile_obj = MidiFile(args_dict['midi_path'])
    return midi_to_features(midi=midifile_obj, max_pairs_number=args_dict['max_pairs_number'])

def main():
    args = parse_args()
    # root logger
    if args.log_file_path != '':
        logging.basicConfig(
            filename=args.log_file_path,
            filemode='a',
            level=logging.INFO,
            format='%(message)s',
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )

    logging.info(strftime('=== get_eval_features_of_dataset.py start at %Y%m%d-%H%M%S ==='))

    if not os.path.isdir(args.dataset_dir_path):
        logging.info('Invalid dataset dir path: %s', args.dataset_dir_path)
        return 1

    file_path_list = glob.glob(args.dataset_dir_path+'/**/*.mid', recursive=True)
    file_path_list += glob.glob(args.dataset_dir_path+'/**/*.midi', recursive=True)
    file_path_list += glob.glob(args.dataset_dir_path+'/**/*.MID', recursive=True)


    dataset_size = len(file_path_list)
    eval_sample_features_per_piece: List[ Dict[str, float] ] = []
    start_time = time()

    while len(eval_sample_features_per_piece) < args.sample_number:
        random_indices = random.sample(list(range(dataset_size)), args.sample_number - len(eval_sample_features_per_piece))
        eval_args_dict_list = [
            {'midi_path': file_path_list[rand_index], 'max_pairs_number': args.max_pairs_number}
            for rand_index in random_indices
        ]
        with Pool(args.workers) as p:
            eval_sample_features += tqdm(
                p.imap(midi_to_features_wrapper, eval_args_dict_list),
                total=args.sample_number
            )
        eval_sample_features_per_piece += [f for f in eval_sample_features if f is not None]

    logging.info(
        'Done. Sampling %d midi files from %s takes %.3f seconds',
        args.sample_number,
        args.dataset_dir_path,
        time() - start_time
    )

    eval_sample_features = {
        fname: [
            fs[fname]
            for fs in eval_sample_features_per_piece
        ]
        for fname in EVAL_FEATURE_NAMES
    }

    eval_sample_features_stats = dict()
    for fname in EVAL_FEATURE_NAMES:
        fname_description = dict(Series(eval_sample_features[fname]).dropna().describe())
        fname_description: Dict[str, np.float64]
        eval_sample_features_stats[fname] = {
            k : float(v) for k, v in fname_description.items()
        }
    with open(os.path.join(args.dataset_dir_path, 'eval_sample_feature_stats.json'), 'w+', encoding='utf8') as eval_stat_file:
        json.dump(eval_sample_features_stats, eval_stat_file)

    logging.info(strftime('=== get_eval_features_of_dataset.py exit ==='))
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
