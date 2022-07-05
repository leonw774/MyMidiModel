import json
import numpy as np
import os
from argparse import ArgumentParser

from midi_util import file_to_paras_and_pieces
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
        'input_file_path',
    )
    parser.add_argument(
        'output_dir_path',
    )

    args = parser.parse_args()

    paras, pieces = file_to_paras_and_pieces(args.input_file_path)
    assert len(pieces) > 0, 'empty corpus'

    if not os.path.isdir(args.output_dir_path):
        os.makedirs(args.output_dir_path)

    vocabs = build_vocabs(pieces, paras, args.max_sample_length)
    vocabs_json_path = os.path.join(args.output_dir_path, 'vocabs.json')
    with open(vocabs_json_path, 'w+', encoding='utf8') as vocabs_file:
        json.dump(vocabs, vocabs_file)

    start_time = time()
    print('Begin write npz')
    # serialize pieces into lists of numpy arrays
    # each is processed from a token and has length of seven:
    #   event, duration, velocity, track_numer, instrument, position, measure_number
    # fill each attribute with the index of the text.
    # if a token doesn't have an attribute of some type, fill the index of PAD instead.
    # if a token has an attrbute, but it is not in the dictionary, fill the index of UNK.
    # measure_number attribute is dynamically determined in training time, so don't fill it now.
    pieces_numpy_list = [text_list_to_numpy(p.split(' '), vocabs) for p in pieces]
    print('Average piece length:', sum(x.shape[0] for x in pieces_numpy_list) / len(pieces_numpy_list))

    # # for debugging
    # np.savetxt('text_list_to_numpy0.txt', pieces_numpy_list[0], fmt='%d')
    # with open('text_list_to_numpy0.txt', 'r', encoding='utf8') as f:
    #     data_list = f.read().split('\n')
    #     text_list = pieces[0].split()
    # data_text_list = [ t + ' '*(max(0, 16-len(t))) + d.replace(' ', '\t') for t, d  in zip(text_list, data_list) ]
    # with open('text_list_to_numpy0.txt', 'w+', encoding='utf8') as f:
    #     f.write('token_text      evt\tdur\tvel\ttrn\tins\ttmp\tpos\tmea\n')
    #     f.write('\n'.join(data_text_list))
    # with open('numpy_to_text_list0.txt', 'w+', encoding='utf8') as f:
    #     f.write('\n'.join(numpy_to_text_list(pieces_numpy_list[0], vocabs, paras['tempo_method'])))

    np.savez_compressed(os.path.join(args.output_dir_path, 'data.npz'), *pieces_numpy_list)
    print('Time:', time()-start_time)
