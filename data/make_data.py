import json
import logging
import numpy as np
import os
from argparse import ArgumentParser
from time import localtime, strftime

from midi_util import file_to_paras_and_pieces_iterator
from data_util import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--max-sample-length',
        dest='max_sample_length',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    parser.add_argument(
        'input_file_path',
    )
    parser.add_argument(
        'output_dir_path',
    )

    loglevel = logging.INFO
    logging.basicConfig(
        filename=strftime("logs/make_data-%Y%m%d-%H%M.log", localtime()),
        filemode='w',
        level=loglevel
    )
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    logging.getLogger().addHandler(console)

    args = parser.parse_args()

    with open(args.input_file_path, 'r', encoding='utf8') as infile:
        paras, piece_iterator = file_to_paras_and_pieces_iterator(infile)
        assert len(piece_iterator) > 0, 'empty corpus'

        if not os.path.isdir(args.output_dir_path):
            os.makedirs(args.output_dir_path)

        vocabs, summary_string = build_vocabs(piece_iterator, paras, args.max_sample_length)
        logging.info(summary_string)
        vocabs_json_path = os.path.join(args.output_dir_path, 'vocabs.json')
        with open(vocabs_json_path, 'w+', encoding='utf8') as vocabs_file:
            json.dump(vocabs, vocabs_file)

        start_time = time()
        logging.info('Begin write npz')
        # serialize pieces into lists of numpy arrays
        # each is processed from a token and has length of seven:
        #   event, duration, velocity, track_numer, instrument, position, measure_number
        # fill each attribute with the index of the text.
        # if a token doesn't have an attribute of some type, fill the index of PAD instead.
        # if a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
        # measure_number attribute is dynamically determined in training time, so don't fill it now.
        pieces_numpy_list = [text_list_to_numpy(p.split(' '), vocabs) for p in piece_iterator]

        # for debugging
        if args.debug:
            np.savetxt('text_list_to_numpy_debug.txt', pieces_numpy_list[0], fmt='%d')
            with open('text_list_to_numpy_debug.txt', 'r', encoding='utf8') as f:
                data_list = f.read().split('\n')
                text_list = next(piece_iterator.__iter__()).split()
            reconst_text_list = numpy_to_text_list(pieces_numpy_list[0], vocabs, paras['tempo_method'])
            text_data_reconst_text_list = [(t, *d.split(), rt) for t, d, rt in zip(text_list, data_list, reconst_text_list)]
            with open('text_list_to_numpy_debug.txt', 'w+', encoding='utf8') as f:
                f.write(
                    f"{'token_text':<14}{'evt':>5}{'dur':>5}{'vel':>5}{'trn':>5}{'ins':>5}{'tmp':>5}{'pos':>5}{'mea':>5} {'reconstructed_text':<14}\n")
                formated_lines = [
                    f"{line[0]:<14}{line[1]:>5}{line[2]:>5}{line[3]:>5}{line[4]:>5}{line[5]:>5}{line[6]:>5}{line[7]:>5}{line[8]:>5} {line[9]:<14}"
                    for line in text_data_reconst_text_list
                ]
                f.write('\n'.join(formated_lines))

    np.savez_compressed(os.path.join(args.output_dir_path, 'data.npz'), *pieces_numpy_list)
    logging.info('npz write time: %.3f', time()-start_time)
    logging.info('make_data.py exited')
