from argparse import ArgumentParser
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
        '--eval-sample-number',
        type=int,
        default=64
    )
    parser.add_argument(
        '--model-dir-path',
        type=str,
        required=True
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
    best_model: MyMidiTransformer = torch.load(os.path.join(args.model_dir_path, 'best_model.pt'))
    vocabs = best_model.vocabs
    uncond_gen_piece_list = []
    uncond_gen_start_time = time()
    uncond_gen_total_token_length = 0
    logging.info('Generating unconditional generation sample for evaluation')
    for _ in tqdm(range(args.eval_sample_number)):
        uncond_gen_text_list = generate_sample(best_model, best_model.max_seq_length)
        uncond_gen_total_token_length += len(uncond_gen_text_list)
        uncond_gen_piece = ' '.join(uncond_gen_text_list)
        uncond_gen_piece_list.append(uncond_gen_piece)
    logging.info(
        'Done. Generating %d pieces with max_length %d takes %.3f seconds',
        args.eval_sample_number,
        best_model.max_seq_length,
        time() - uncond_gen_start_time
    )
    logging.info('Avg. tokens# in the samples are %.3f', uncond_gen_total_token_length / args.eval_sample_number)

    logging.info('Dumping unconditional generation sample to %s', eval_dir_path)
    eval_sample_features_per_piece = []
    eval_sample_features_per_piece: List[ Dict[str, float] ]
    for i, uncond_gen_piece in enumerate(uncond_gen_piece_list):
        open(os.path.join(eval_dir_path, f'{i}.txt'), 'w+', encoding='utf8').write(uncond_gen_piece)
        try:
            piece_to_midi(uncond_gen_piece, vocabs.paras['nth'], ignore_pending_note_error=True).dump(
                os.path.join(eval_dir_path, f'{i}.mid')
            )
            eval_sample_features_per_piece.append(
                piece_to_features(uncond_gen_piece, nth=vocabs.paras['nth'], max_pairs_number=int(10e6))
            )
        except (AssertionError, ValueError):
            print('Error when dumping eval uncond gen MidiFile object')
            print(format_exc())

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
    with open(os.path.join(args.model_dir_path, 'eval_sample_feature_stats.json'), 'w+', encoding='utf8') as eval_stat_file:
        json.dump(eval_sample_features_stats, eval_stat_file)


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
