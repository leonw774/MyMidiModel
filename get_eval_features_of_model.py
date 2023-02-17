from argparse import ArgumentParser
from collections import Counter
import json
import logging
import os
from time import strftime, time
from traceback import format_exc
from typing import List, Dict

import numpy as np
from pandas import Series
import torch
from tqdm import tqdm

from util.midi import piece_to_midi
from util.model import MyMidiTransformer, generate_sample
from util.evaluations import EVAL_SCALAR_FEATURE_NAMES, EVAL_DISTRIBUTION_FEATURE_NAMES, piece_to_features, _kl_divergence

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
        '--max-pairs-number',
        type=int,
        default=int(1e6)
    )
    parser.add_argument(
        '--data-dir-path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--use-device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        'model_dir_path',
        type=str
    )
    return parser.parse_args()


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

    logging.info(strftime('=== get_eval_features_of_model.py start at %Y%m%d-%H%M%S ==='))

    eval_dir_path = os.path.join(args.model_dir_path, 'eval_samples')
    if not os.path.isdir(eval_dir_path):
        logging.info('Invalid model eval samples dir path: %s', eval_dir_path)
        return 1

    logging.info('Loading model at %s', os.path.join(args.model_dir_path, 'best_model.pt'))

    if not args.use_device.startswith('cuda') and args.use_device != 'cpu':
        raise ValueError(f'Bad device name {args.use_device}')
    if not torch.cuda.is_available():
        args.use_device = 'cpu'

    best_model: MyMidiTransformer = torch.load(os.path.join(args.model_dir_path, 'best_model.pt'), map_location=torch.device(args.use_device))
    
    vocabs = best_model.vocabs
    eval_features_per_piece = []
    eval_features_per_piece: List[ Dict[str, float] ]
    uncond_gen_start_time = time()
    uncond_gen_total_token_length = 0
    logging.info('Generating unconditional generation sample for evaluation')
    for i in tqdm(range(args.sample_number)):
        # generate
        uncond_gen_text_list = generate_sample(best_model, best_model.max_seq_length)
        uncond_gen_total_token_length += len(uncond_gen_text_list)
        uncond_gen_piece = ' '.join(uncond_gen_text_list)

        # save to disk
        open(os.path.join(eval_dir_path, f'{i}.txt'), 'w+', encoding='utf8').write(uncond_gen_piece)
        try:
            piece_to_midi(uncond_gen_piece, vocabs.paras['nth']).dump(
                os.path.join(eval_dir_path, f'{i}.mid')
            )
            eval_features_per_piece.append(
                piece_to_features(uncond_gen_piece, nth=vocabs.paras['nth'], max_pairs_number=args.max_pairs_number)
            )
        except (AssertionError, ValueError):
            print(f'Error when dumping eval uncond gen #{i} MidiFile object')
            print(format_exc())

    logging.info(
        'Done. Generating %d pieces with max_length %d takes %.3f seconds',
        args.sample_number,
        best_model.max_seq_length,
        time() - uncond_gen_start_time
    )
    logging.info('Avg. tokens# in the samples are %.3f', uncond_gen_total_token_length / args.sample_number)

    aggr_scalar_eval_features = {
        fname: [
            features[fname]
            for features in eval_features_per_piece
        ]
        for fname in EVAL_SCALAR_FEATURE_NAMES
    }

    model_eval_features_stats = dict()
    for fname in EVAL_SCALAR_FEATURE_NAMES:
        fname_description = dict(Series(aggr_scalar_eval_features[fname]).dropna().describe())
        fname_description: Dict[str, np.float64]
        model_eval_features_stats[fname] = {
            k : float(v) for k, v in fname_description.items()
        }

    for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
        model_eval_features_stats[fname] = dict(sum(
            [Counter(features[fname]) for features in eval_features_per_piece],
            Counter() # starting value of empty counter
        ))

    if args.dataset_dir_path != '':
        dataset_eval_stat_file_path = os.path.join(args.dataset_dir_path, 'eval_feature_stats.json')
        if os.path.isfile(dataset_eval_stat_file_path):
            with open(dataset_eval_stat_file_path, 'r', encoding='utf8') as dataset_eval_stat_file:
                dataset_eval_features_stats = json.load(dataset_eval_stat_file)
            logging.info(
                'Computing KL-divergance of pitch, duration, and velocity between %s and %s',
                args.dataset_dir_path,
                args.model_dir_path
            )
            for fname in EVAL_DISTRIBUTION_FEATURE_NAMES:
                model_eval_features_stats[fname+'_KLD'] = _kl_divergence(
                    dataset_eval_features_stats[fname],
                    model_eval_features_stats[fname]
                )
        else:
            logging.info('Path: %s does not have "eval_feature_stats.json"', args.dataset_dir_path)

    with open(os.path.join(args.model_dir_path, 'eval_feature_stats.json'), 'w+', encoding='utf8') as eval_stat_file:
        json.dump(model_eval_features_stats, eval_stat_file)

    logging.info(strftime('=== get_eval_features_of_model.py exit ==='))


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
