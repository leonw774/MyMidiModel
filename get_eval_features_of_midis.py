from argparse import ArgumentParser
from collections import Counter
import glob
import json
import logging
from multiprocessing import Pool
import os
from psutil import cpu_count
from time import strftime, time
from traceback import format_exc
from typing import List, Dict
import random

from miditoolkit import MidiFile
import numpy as np
from pandas import Series
from tqdm import tqdm

from util.evaluations import EVAL_SCALAR_FEATURE_NAMES, EVAL_DISTRIBUTION_FEATURE_NAMES, midi_to_features, kl_divergence

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
        default=min(cpu_count(), 4)
    )
    parser.add_argument(
        '--max-pairs-number',
        type=int,
        default=int(1e6)
    )
    parser.add_argument(
        '--reference-file-path',
        type=str,
        default='',
        help='Should set to a path to a reference result JSON file (Q).\
            If is set, the result of MIDI_DIR_PATH (P) will include KL divergence of pitch, duration, and velocity\
            of MIDI_DIR_PATH from the those in Q (That is: KL(P||Q))'
    )
    parser.add_argument(
        'midi_dir_path',
        type=str,
        help='Find all midi files under this directory recursively,\
            and output the result JSON file (\"eval_feature_stats.json\") at this path.'
    )
    return parser.parse_args()


def midi_to_features_wrapper(args_dict: dict):
    try:
        midifile_obj = MidiFile(args_dict['midi_file_path'])
        features = midi_to_features(midi=midifile_obj, max_pairs_number=args_dict['max_pairs_number'])
    except Exception:
        logging.debug(format_exc())
        return None
    return features

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

    logging.info(strftime('=== get_eval_features_of_midis.py start at %Y%m%d-%H%M%S ==='))

    if not os.path.isdir(args.midi_dir_path):
        logging.info('Invalid dir path: %s', args.midi_dir_path)
        return 1

    file_path_list = glob.glob(args.midi_dir_path+'/**/*.mid', recursive=True)
    file_path_list += glob.glob(args.midi_dir_path+'/**/*.midi', recursive=True)
    file_path_list += glob.glob(args.midi_dir_path+'/**/*.MID', recursive=True)

    dataset_size = len(file_path_list)
    assert dataset_size > 0, f'No midi files found in {args.midi_dir_path}'
    if dataset_size < args.sample_number:
        logging.info('Dataset size (%d) is smaller than given sample number (%d)', dataset_size, args.sample_number)
        args.sample_number = dataset_size
    eval_features_per_piece: List[ Dict[str, float] ] = []
    start_time = time()

    while len(eval_features_per_piece) < args.sample_number:
        if args.sample_number >= dataset_size:
            sampled_indices = list(range(dataset_size))
        else:
            sampled_indices = random.sample(list(range(dataset_size)), args.sample_number - len(eval_features_per_piece))
        eval_args_dict_list = [
            {'midi_file_path': file_path_list[idx], 'max_pairs_number': args.max_pairs_number}
            for idx in sampled_indices
        ]
        with Pool(args.workers) as p:
            eval_features = list(tqdm(
                p.imap_unordered(midi_to_features_wrapper, eval_args_dict_list),
                total=len(sampled_indices)
            ))
            eval_features = [f for f in eval_features if f is not None]
            print(f'Processed {len(eval_features)} uncorrupted files out of {len(sampled_indices)} random indices')
            eval_features_per_piece += eval_features

    logging.info(
        'Done. Sampling %d midi files from %s takes %.3f seconds',
        args.sample_number,
        args.midi_dir_path,
        time() - start_time
    )

    aggr_scalar_eval_features = {
        fname: [
            fs[fname]
            for fs in eval_features_per_piece
        ]
        for fname in EVAL_SCALAR_FEATURE_NAMES
    }

    eval_features_stats = dict()
    for fname in EVAL_SCALAR_FEATURE_NAMES:
        fname_description = dict(Series(aggr_scalar_eval_features[fname]).dropna().describe())
        fname_description: Dict[str, np.float64]
        eval_features_stats[fname] = {
            k : float(v) for k, v in fname_description.items()
        }

    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        eval_features_stats[fname] = dict(sum(
            [Counter(features[fname]) for features in eval_features_per_piece],
            Counter() # starting value of empty counter
        ))

    if os.path.isfile(args.reference_file_path):
        with open(args.reference_file_path, 'r', encoding='utf8') as reference_file:
            try:
                reference_eval_features_stats = json.load(reference_file)
            except Exception:
                logging.info('json.load(%s) failed', args.reference_file_path)
                logging.info(format_exc())
                return 1
        logging.info(
            'Computing KL-divergance of pitch, duration, and velocity of midis in %s from %s',
            args.midi_dir_path,
            args.reference_file_path
        )
        for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
            eval_features_stats[fname+'_KLD'] = kl_divergence(
                eval_features_stats[fname],
                reference_eval_features_stats[fname]
            )
    else:
        logging.info('%s is invalid path forreference result JSON file', args.reference_file_path)

    eval_stat_file_path = os.path.join(args.midi_dir_path, 'eval_feature_stats.json')
    with open(eval_stat_file_path, 'w+', encoding='utf8') as eval_stat_file:
        json.dump(eval_features_stats, eval_stat_file)
        logging.info('Outputed evaluation feature stats JSON at %s', eval_stat_file_path)

    logging.info(strftime('=== get_eval_features_of_midis.py exit ==='))
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
