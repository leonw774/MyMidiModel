from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
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
from mido import MidiFile as mido_MidiFile
import numpy as np
from pandas import Series
from tqdm import tqdm

from util.corpus import get_corpus_paras
from util.evaluations import (
    EVAL_SCALAR_FEATURE_NAMES,
    EVAL_DISTRIBUTION_FEATURE_NAMES,
    midi_to_features,
    kl_divergence,
    overlapping_area_of_estimated_gaussian,
    histogram_intersection
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--sample-number',
        type=int,
        default=-1,
        help='The number of random selected sample to be use to evaluate.\
            If it is set to -1, all the files would be used.\
                Default is %(default)s.'
    )
    parser.add_argument(
        '--midi-to-piece-paras',
        type=str,
        default='',
        help='The path of midi_to_piece parameters file (the YAML file).\
            Default is empty string and `util.evaluations.EVALUATION_MIDI_TO_PIECE_PARAS_DEFAULT` will be used.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=min(cpu_count(), 4)
    )
    parser.add_argument(
        '--primer-measure-length',
        type=int,
        default=0,
        metavar='k',
        help='If this option is not set, the features are computed from the whole piece.\
            If this option is set to %(metavar)s, which should be a positive integer,\
            the features are computed from the number %(metavar)s+1 to the last measure of the piece.'
    )
    parser.add_argument(
        '--log',
        dest='log_file_path',
        type=str,
        default='',
    )
    parser.add_argument(
        '--output-sampled-file-paths',
        action='store_true',
        help='If set, the paths of the sampled files would be output to \"MIDI_DIR_PATH/eval_pathlist.txt\"'
    )
    parser.add_argument(
        '--max-pairs-number',
        type=int,
        default=int(1e6),
        help='Maximal limit of measure pairs for calculating the self-similarities. Default is %(default)s.'
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
        '--seed',
        type=int,
        default=None
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
        features = midi_to_features(
            midi=midifile_obj,
            primer_measure_length=args_dict['primer_measure_length'],
            max_pairs_number=args_dict['max_pairs_number']
        )
    except Exception:
        # print(format_exc())
        return None
    return features

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
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

    logging.info(strftime('==== get_eval_features_of_midis.py start at %Y%m%d-%H%M%S ===='))

    if not os.path.isdir(args.midi_dir_path):
        logging.info('Invalid dir path: %s', args.midi_dir_path)
        return 1

    file_path_list = glob.glob(args.midi_dir_path+'/**/*.mid', recursive=True)
    file_path_list += glob.glob(args.midi_dir_path+'/**/*.midi', recursive=True)
    file_path_list += glob.glob(args.midi_dir_path+'/**/*.MID', recursive=True)

    dataset_size = len(file_path_list)
    assert dataset_size > 0, f'No midi files found in {args.midi_dir_path}'
    if args.sample_number == -1:
        logging.info('Using all (%d) midis in the dataset', dataset_size)
        args.sample_number = dataset_size
    elif dataset_size < args.sample_number:
        logging.info('Dataset size (%d) is smaller than given sample number (%d)', dataset_size, args.sample_number)
        args.sample_number = dataset_size

    if args.midi_to_piece_paras != '':
        assert os.path.isfile(args.midi_to_piece_paras), f'{args.midi_to_piece_paras} doesn\'t exist or is not a file'
        paras_dict = get_corpus_paras(args.midi_to_piece_paras)
    else:
        paras_dict = None

    args.workers = min(args.workers, dataset_size)

    sampled_midis_eval_features: List[ Dict[str, float] ] = []
    sample_able_indices = set(range(dataset_size))
    sampled_midi_file_paths = []
    sampled_midi_length = []

    start_time = time()
    while len(sampled_midis_eval_features) < args.sample_number and len(sample_able_indices) > 0:
        if args.sample_number >= len(sample_able_indices):
            random_indices = list(sample_able_indices)
            sample_able_indices.clear()
        else:
            random_indices = random.sample(list(sample_able_indices), args.sample_number - len(sampled_midis_eval_features))
            sample_able_indices.difference_update(random_indices)

        eval_args_dict_list = [
            {
                'midi_file_path': file_path_list[idx],
                'midi_to_piece_paras': paras_dict,
                'primer_measure_length': args.primer_measure_length,
                'max_pairs_number': args.max_pairs_number
            }
            for idx in random_indices
        ]
        with Pool(args.workers) as p:
            eval_features = list(tqdm(
                p.imap(midi_to_features_wrapper, eval_args_dict_list),
                total=len(random_indices)
            ))
        uncorrupt_midi_file_paths = [
            file_path_list[idx]
            for n, idx in enumerate(random_indices)
            if eval_features[n] is not None
        ]
        eval_features = [f for f in eval_features if f is not None]
        sampled_midis_eval_features += eval_features
        sampled_midi_file_paths += uncorrupt_midi_file_paths
        sampled_midi_length += [
            mido_MidiFile(midi_path).length
            for midi_path in uncorrupt_midi_file_paths
        ]
        print(f'Processed {len(eval_features)} uncorrupted files out of {len(random_indices)} random indices')

    logging.info(
        'Done. Sampling %d midi files from %s takes %.3f seconds.',
        len(sampled_midis_eval_features),
        args.midi_dir_path,
        time() - start_time
    )

    eval_features_stats = dict()
    # process scalar features
    aggr_scalar_eval_features = {
        fname: [
            fs[fname]
            for fs in sampled_midis_eval_features
        ]
        for fname in EVAL_SCALAR_FEATURE_NAMES
    }
    for fname in EVAL_SCALAR_FEATURE_NAMES:
        fname_description = dict(Series(aggr_scalar_eval_features[fname], dtype='float64').dropna().describe())
        fname_description: Dict[str, np.float64]
        eval_features_stats[fname] = {
            k: float(v) for k, v in fname_description.items()
        }
    # process distribution features
    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        eval_features_stats[fname] = dict(sum(
            [Counter(features[fname]) for features in sampled_midis_eval_features],
            Counter() # starting value of empty counter
        ))
        # cast the keys into strings!!! because the reference distribution is read from json, and their keys are strings
        eval_features_stats[fname] = {str(k): v for k, v in eval_features_stats[fname].items()}
    # process other
    eval_features_stats['notes_number_per_piece'] = [
        sum(features['pitch_histogram'].values())
        for features in sampled_midis_eval_features
    ]
    logging.info(
        '%d notes involved in evaluation. Avg. #note per piece: %f. Tot. midi playback time: %f.',
        np.sum(eval_features_stats['notes_number_per_piece']),
        np.mean(eval_features_stats['notes_number_per_piece']),
        np.sum(sampled_midi_length)
    )

    logging.info('\t'.join([
        f'{fname}' for fname in EVAL_SCALAR_FEATURE_NAMES
    ]))
    logging.info('\t'.join([
        f'{eval_features_stats[fname]["mean"]}' for fname in EVAL_SCALAR_FEATURE_NAMES
    ]))

    if args.reference_file_path != '':
        if os.path.isfile(args.reference_file_path):
            with open(args.reference_file_path, 'r', encoding='utf8') as reference_file:
                try:
                    reference_eval_features_stats = json.load(reference_file)
                except Exception:
                    logging.info('json.load(%s) failed', args.reference_file_path)
                    logging.info(format_exc())
                    return 1
            logging.info(
                'Computing KLd, OA, and HI of pitch, duration, and velocity of midis in %s from %s',
                args.midi_dir_path,
                args.reference_file_path
            )

            # KL Divergence
            for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
                eval_features_stats[fname+'_KLD'] = kl_divergence(
                    eval_features_stats[fname],
                    reference_eval_features_stats[fname],
                    ignore_pred_zero=True
                )

            # Overlapping area of estimated gaussian distribution
            for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
                # because the keys are strings and are represented as interger (pitch & velocity) and fraction (duration)
                eval_features_stats[fname+'_OA'] = overlapping_area_of_estimated_gaussian(
                    {float(Fraction(k)): v for k, v in eval_features_stats[fname].items()},
                    {float(Fraction(k)): v for k, v in reference_eval_features_stats[fname].items()}
                )

            # (Normalized) Histogram intersection
            for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
                eval_features_stats[fname+'_HI'] = histogram_intersection(
                    eval_features_stats[fname],
                    reference_eval_features_stats[fname]
                )

            logging.info('\t'.join([
                f'{fname}{suffix}'
                for suffix in ('_KLD', '_OA', '_HI')
                for fname in EVAL_DISTRIBUTION_FEATURE_NAMES
            ]))

            logging.info('\t'.join([
                f'{eval_features_stats[fname+suffix]}'
                for suffix in ('_KLD', '_OA', '_HI')
                for fname in EVAL_DISTRIBUTION_FEATURE_NAMES
            ]))

        else:
            logging.info('%s is invalid path for reference result JSON file', args.reference_file_path)

    eval_feat_file_path = os.path.join(args.midi_dir_path, 'eval_features.json')
    with open(eval_feat_file_path, 'w+', encoding='utf8') as eval_feat_file:
        json.dump(eval_features_stats, eval_feat_file)
        logging.info('Outputed evaluation features JSON at %s', eval_feat_file_path)

    if args.output_sampled_file_paths:
        eval_pathlist_file_path = os.path.join(args.midi_dir_path, 'eval_pathlist.txt')
        with open(eval_pathlist_file_path, 'w+', encoding='utf8') as eval_pathlist_file:
            eval_pathlist_file.write('\n'.join(sampled_midi_file_paths)+'\n')
            logging.info('Outputed evaluation sampled file paths at %s', eval_pathlist_file_path)

    logging.info(strftime('==== get_eval_features_of_midis.py exit ===='))
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
