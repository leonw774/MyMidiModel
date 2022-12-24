from argparse import ArgumentParser
import json
import logging
import os
from time import strftime, time
from traceback import format_exc
from typing import List, Dict
import random

import numpy as np
from pandas import Series
from tqdm import tqdm

from util.corpus import get_corpus_vocabs, CorpusReader
from util.evaluations import EVAL_FEATURE_NAMES, piece_to_features

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
        default=64
    )
    parser.add_argument(
        'corpus_dir_path',
        type=str
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # root logger
    # if args.log_file_path != '':
    #     logging.basicConfig(
    #         filename=args.log_file_path,
    #         filemode='a',
    #         level=logging.INFO,
    #         format='%(message)s',
    #     )
    #     console = logging.StreamHandler()
    #     console.setLevel(logging.INFO)
    #     logging.getLogger().addHandler(console)
    # else:
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='%(message)s'
    #     )

    logging.info(strftime('=== get_eval_features_of_dataset.py start at %Y%m%d-%H%M%S ==='))

    if not os.path.isdir(args.corpus_dir_path):
        logging.info('Invalid dataset dir path: %s', args.corpus_dir_path)
        return 1

    vocabs = get_corpus_vocabs(args.corpus_dir_path)
    with CorpusReader(args.corpus_dir_path) as corpus_reader:
        corpus_len = len(corpus_reader)
        eval_sample_features_per_piece = []
        eval_sample_features_per_piece: List[ Dict[str, float] ]
        start_time = time()
        random_piece_total_token_length = 0
        logging.info('Generating unconditional generation sample for evaluation')
        sampled_rand_index = set()
        for i in tqdm(range(args.sample_number)):
            # get random piece
            while True:
                rand_index = random.randint(corpus_len)
                if rand_index not in sampled_rand_index:
                    break
            random_piece = corpus_reader[rand_index]
            random_piece_total_token_length += len(random_piece.count(' ') + 1)

            try:
                eval_sample_features_per_piece.append(
                    piece_to_features(random_piece, nth=vocabs.paras['nth'], max_pairs_number=int(10e6))
                )
            except (AssertionError, ValueError):
                print(f'Error when getting feature from piece witat index #{rand_index} MidiFile object')
                print(format_exc())

    logging.info(
        'Done. Sampling %d pieces from %s takes %.3f seconds',
        args.eval_sample_number,
        args.corpus_dir_path,
        time() - start_time
    )
    logging.info('Avg. tokens# in the samples are %.3f', random_piece_total_token_length / args.eval_sample_number)

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
    with open(os.path.join(args.corpus_dir_path, 'eval_sample_feature_stats.json'), 'w+', encoding='utf8') as eval_stat_file:
        json.dump(eval_sample_features_stats, eval_stat_file)

    logging.info(strftime('=== get_eval_features_of_dataset.py exit ==='))


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
