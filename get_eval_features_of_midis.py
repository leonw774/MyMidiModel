from argparse import ArgumentParser
import glob
import json
import logging
import os
import random
from time import strftime, time
from traceback import format_exc

from miditoolkit import MidiFile
from mido import UnknownMetaMessage, MidiFile as mido_MidiFile
import numpy as np
from psutil import cpu_count
from tqdm import tqdm

from util.corpus import get_corpus_paras
from util.evaluations import (
    EVAL_SCALAR_FEATURE_NAMES,
    EVAL_DISTRIBUTION_FEATURE_NAMES,
    midi_list_to_features,
    compare_with_ref
)
from util.type_wrappers import or_none


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--midi-to-piece-paras',
        type=str,
        nargs='?',
        const='',
        default='',
        help='The path of midi_to_piece parameters file (the YAML file).\
            Default is empty string and \
            `util.evaluations.EVALUATION_MIDI_TO_PIECE_PARAS_DEFAULT` is used.'
    )
    parser.add_argument(
        '--worker-number',
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
            the features are computed from the number %(metavar)s+1 to the last measure.'
    )
    parser.add_argument(
        '--log',
        dest='log_file_path',
        type=str,
        default='',
    )
    parser.add_argument(
        '--max-pairs-number',
        type=int,
        default=int(1e6),
        help='Maximal limit of measure pairs for calculating the self-similarities. \
            Default is %(default)s.'
    )
    parser.add_argument(
        '--reference-file-path',
        type=str,
        default='',
        help='Should set to a path to a reference result JSON file.'
    )
    parser.add_argument(
        '--seed',
        type=or_none(int),
        default=None
    )
    parser.add_argument(
        'midi_dir_path',
        type=str,
        help='Find all midi files under this directory recursively,\
            and output the result JSON file (\"eval_features.json\") at this path.'
    )
    return parser.parse_args()



def fix_typeerror_from_msg_copy(mido_midi: mido_MidiFile):
    for i, track in enumerate(mido_midi.tracks):
        mido_midi.tracks[i] = [
            msg
            for msg in track
            if msg.__class__ != UnknownMetaMessage
        ]


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

    midi_path_list = glob.glob(args.midi_dir_path+'/**/*.mid', recursive=True)
    midi_path_list += glob.glob(args.midi_dir_path+'/**/*.midi', recursive=True)
    midi_path_list += glob.glob(args.midi_dir_path+'/**/*.MID', recursive=True)

    dataset_size = len(midi_path_list)
    assert dataset_size > 0, f'No midi files found in {args.midi_dir_path}'
    logging.info('Found total (%d) midis in %s', dataset_size, args.midi_dir_path)

    if args.midi_to_piece_paras != '':
        assert os.path.isfile(args.midi_to_piece_paras), \
            f'{args.midi_to_piece_paras} doesn\'t exist or is not a file'
        paras_dict = get_corpus_paras(args.midi_to_piece_paras)
    else:
        paras_dict = None

    args.worker_number = min(args.worker_number, dataset_size)

    # filter out midi paths that can not processed by miditoolkit or mido
    midi_list = []
    new_midi_path_list = []
    for p in midi_path_list:
        try:
            midi_list.append(MidiFile(p))
            new_midi_path_list.append(p)
        except Exception:
            pass
    midi_path_list = new_midi_path_list

    start_time = time()
    uncorrupt_midi_indices_list, aggr_eval_features = midi_list_to_features(
        midi_list=midi_list,
        midi_to_piece_paras=paras_dict,
        primer_measure_length=args.primer_measure_length,
        max_pairs_number=args.max_pairs_number,
        worker_number=args.worker_number,
        use_tqdm=True
    )
    logging.info(
        'Processed %d uncorrupted files out of %d takes %.3f seconds.',
        len(uncorrupt_midi_indices_list),
        dataset_size,
        time() - start_time
    )

    # compute midi durations
    midi_length_list = []
    for i in tqdm(uncorrupt_midi_indices_list, desc='Getting MIDI durations', ncols=100):
        midi_path = midi_path_list[i]
        midomidi = mido_MidiFile(midi_path)
        try:
            midi_dur = midomidi.length
            midi_length_list.append(midi_dur)
        except TypeError:
            # stupid bug
            fix_typeerror_from_msg_copy(midomidi)
            midi_dur = midomidi.length
            midi_length_list.append(midi_dur)
    logging.info(
        '%d notes involved in evaluation. Avg. #note per piece: %f. Tot. midi playback time: %f.',
        np.sum(aggr_eval_features['notes_number_per_piece']),
        np.mean(aggr_eval_features['notes_number_per_piece']),
        np.sum(midi_length_list)
    )

    logging.info('\t'.join([
        f'{fname}' for fname in EVAL_SCALAR_FEATURE_NAMES
    ]))
    logging.info('\t'.join([
        f'{aggr_eval_features[fname]["mean"]}' for fname in EVAL_SCALAR_FEATURE_NAMES
    ]))

    if args.reference_file_path != '':
        if os.path.isfile(args.reference_file_path):
            with open(args.reference_file_path, 'r', encoding='utf8') as reference_file:
                try:
                    reference_eval_features = json.load(reference_file)
                except Exception:
                    logging.info('json.load(%s) failed', args.reference_file_path)
                    logging.info(format_exc())
                    return 1

            compare_with_ref(aggr_eval_features, reference_eval_features)

            logging.info('\t'.join([
                f'{fname}{suffix}'
                for suffix in ('_KLD', '_OA', '_HI')
                for fname in EVAL_DISTRIBUTION_FEATURE_NAMES
            ]))

            logging.info('\t'.join([
                f'{aggr_eval_features[fname+suffix]}'
                for suffix in ('_KLD', '_OA', '_HI')
                for fname in EVAL_DISTRIBUTION_FEATURE_NAMES
            ]))

        else:
            logging.info(
                '%s is invalid path for reference result JSON file',
                args.reference_file_path
            )

    eval_feat_file_path = os.path.join(args.midi_dir_path, 'eval_features.json')
    with open(eval_feat_file_path, 'w+', encoding='utf8') as eval_feat_file:
        json.dump(aggr_eval_features, eval_feat_file)
    logging.info('Outputed evaluation features JSON at %s', eval_feat_file_path)

    logging.info(strftime('==== get_eval_features_of_midis.py exit ===='))
    return 0


if __name__ == '__main__':
    EXIT_CODE = main()
    exit(EXIT_CODE)
